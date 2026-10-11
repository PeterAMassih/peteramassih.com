# patches/mask2former/viz/data.py
"""Run Mask2Former once on the dog photo and cache what the scenes draw.

The Manim scenes show the real model, but rendering should not need torch, so
this script saves everything they use to sources/dog.npz (not committed):

- logits: the dog query's final mask logits at stride 4, padding removed
- layer_logits: the same query's mask logits from all ten supervised outputs,
  the learnable queries' prediction first, then one per decoder layer
- layer_p_dog: the dog probability at each of those ten outputs
- attn_l: the query's cross-attention weights in decoder layer l (1 to 9),
  averaged over heads, on that layer's feature grid with padding removed
- gate_l: the locations layer l was allowed to read, rebuilt the way the
  model builds them: previous mask logits resized bilinearly (torch, no
  antialiasing) to the layer's grid, then sigmoid >= 0.5
- q_hat, q_norm: the direction and length of the dog query's final mask
  embedding, the vector the mask head dots with every pixel embedding
- x_map, y_map: each pixel embedding projected onto q_hat, and onto the
  widest-spread direction perpendicular to it. Since the mask logit is the
  dot product with no bias, q_norm * x_map is exactly the final logit map.

Cross-attention weights and mask embeddings are not returned by the Hugging
Face model, so hooks keep them.

Run (from this directory): .venv/bin/python data.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from fig_mask_classification import CKPT, HERE, load_image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

OUT = HERE / "sources" / "dog.npz"


def main():
    img = load_image()
    proc = AutoImageProcessor.from_pretrained(CKPT, size={"shortest_edge": 800, "longest_edge": 1333})
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT).eval()
    inputs = proc(images=img, return_tensors="pt")

    attn, mask_embeds = [], []
    decoder = model.model.transformer_module.decoder
    for layer in decoder.layers:
        layer.cross_attn.register_forward_hook(lambda m, i, o: attn.append(o[1][0]))
    # Called once per supervised output. The last call feeds the final logits.
    decoder.mask_predictor.mask_embedder.register_forward_hook(lambda m, i, o: mask_embeds.append(o[0]))
    with torch.no_grad():
        out = model(**inputs, output_auxiliary_logits=True)

    dog = next(k for k, v in model.config.id2label.items() if v == "dog")
    q = int(torch.softmax(out.class_queries_logits[0], -1)[:, dog].argmax())
    outputs = out.auxiliary_logits + [{"masks_queries_logits": out.masks_queries_logits,
                                       "class_queries_logits": out.class_queries_logits}]

    # The input may be padded to a multiple of 32 on the bottom and right. Keep the image part.
    valid = inputs["pixel_mask"][0].numpy().astype(bool)
    img_h, img_w = valid.any(1).sum(), valid.any(0).sum()
    pad_h, pad_w = inputs["pixel_values"].shape[-2:]
    h4, w4 = img_h // 4, img_w // 4

    data = {
        "logits": outputs[-1]["masks_queries_logits"][0, q, :h4, :w4].numpy(),
        "layer_logits": np.stack([o["masks_queries_logits"][0, q, :h4, :w4].numpy() for o in outputs]),
        "layer_p_dog": np.array([float(torch.softmax(o["class_queries_logits"][0, q], -1)[dog]) for o in outputs]),
    }
    for l, w in enumerate(attn, start=1):
        stride = (32, 16, 8)[(l - 1) % 3]
        size = (-(-pad_h // stride), -(-pad_w // stride))
        keep = (slice(0, -(-img_h // stride)), slice(0, -(-img_w // stride)))
        data[f"attn_{l}"] = w[q].reshape(size).numpy()[keep]
        prev = outputs[l - 1]["masks_queries_logits"][0, q][None, None]
        gate = F.interpolate(prev, size=size, mode="bilinear", align_corners=False)[0, 0].sigmoid() >= 0.5
        data[f"gate_{l}"] = gate.numpy()[keep]
        assert data[f"attn_{l}"][~data[f"gate_{l}"]].sum() < 1e-6, f"layer {l} reads outside its gate"

    m = mask_embeds[-1][q].double().numpy()
    pix = out.pixel_decoder_last_hidden_state[0, :, :h4, :w4].double().numpy()
    q_norm = np.linalg.norm(m)
    q_hat = m / q_norm
    x_map = np.tensordot(q_hat, pix, axes=1)
    rest = pix.reshape(len(m), -1) - np.outer(q_hat, x_map.ravel())
    u = np.linalg.svd(rest - rest.mean(1, keepdims=True), full_matrices=False)[0][:, 0]
    u -= (u @ q_hat) * q_hat
    u /= np.linalg.norm(u)
    y_map = np.tensordot(u, pix, axes=1)
    # SVD's sign is arbitrary. Putting the dog's pixels above the background makes reruns agree.
    if y_map[x_map > 0].mean() < y_map[x_map <= 0].mean():
        y_map = -y_map
    gap = np.abs(q_norm * x_map - data["logits"]).max()
    assert gap < 1e-3, f"the projection misses the logits by {gap}"
    data.update(q_hat=q_hat.astype(np.float32), q_norm=np.float32(q_norm),
                x_map=x_map.astype(np.float32), y_map=y_map.astype(np.float32))
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT}: query {q}, p(dog) per output {np.round(data['layer_p_dog'], 3)}")


if __name__ == "__main__":
    main()
