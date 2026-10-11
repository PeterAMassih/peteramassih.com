---
title: "Mask2Former, Dissected"
description: "A ground-up walk through Mask2Former, from set prediction and masked attention to point-sampled losses and the ablation results."
pubDate: 2026-07-11
tags: [computer-vision, segmentation, transformers, paper-dissection]
math: true
---

**In short.** Image segmentation labels every pixel of an image. There are three versions of the task, semantic, instance and panoptic (§1). By late 2021, universal models, built to handle all three, did well on semantic and panoptic segmentation. But on instance segmentation, which separates individual objects, they still lost to specialized systems. MaskFormer, Mask2Former's direct predecessor, trailed the best of them by over 9 AP, the standard instance score. Mask2Former [[Cheng et al. 2022](#ref-cheng2022)] kept MaskFormer's core idea, one mask and one class per learned query, and rebuilt its decoder and its training recipe. It closed the gap, at 50.1 AP against 49.5 for the specialized Swin-HTC++, and reported the best results to date on the other two versions (§8).

**How it works, in five steps.**

1. The model keeps a fixed set of learned vectors called queries, 100 by default. Each query predicts one region of the image as a mask, and what that region is as a class (§2).
2. A query makes its mask by taking the dot product of its vector with the vector of every pixel. A high value means the pixel belongs to the query's region (§3).
3. In training, each real object is paired with exactly one query, which learns to predict it. A pair's cost measures how badly the query's class and mask fit the object, and the pairing with the lowest total cost wins. Unpaired queries learn to predict "no object" (§4).
4. Nine decoder layers refine the queries. In each layer, a query reads image features only inside the mask it predicted at the previous layer, so the background can no longer drown out its object. This is masked attention, the paper's main idea (§5).
5. Two more ideas make this affordable. The layers take turns reading coarse and fine feature maps, and the mask losses are computed on 12,544 sampled points instead of every pixel, which cuts training memory from 18 GB to 6 GB per image (§6).

Steps 1 to 3 come from MaskFormer, and steps 4 and 5 are new. Together they form a loop: every layer's mask is trained against the true mask, and each trained mask decides where the next layer looks.

<figure class="viz" id="overview">
<svg class="m2f-arch" style="min-width: 560px" viewBox="0 0 720 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mask2Former in one picture. The backbone and pixel decoder turn the image into features at three resolutions and a per-pixel embedding map. N learned queries read those features in a nine-layer Transformer decoder. Each query predicts a class and a mask, and each mask limits where the next layer looks.">
<defs><style>
.ar-h { font-family: var(--font-sans); fill: var(--color-text); font-size: 13px; font-weight: 600; }
.ar-t { font-family: var(--font-sans); fill: var(--color-text); font-size: 12px; }
.ar-s { font-family: var(--font-sans); fill: var(--color-text-muted); font-size: 11px; }
.ar-g { font-family: var(--font-sans); fill: var(--viz-gold-text); font-size: 12px; }
</style><marker id="ar-ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" style="fill: var(--viz-line)"/></marker><marker id="ar-ag" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" style="fill: var(--viz-gold)"/></marker></defs>
<rect width="720" height="280" rx="6" style="fill: var(--color-bg)"/>
<rect x="16" y="52" width="70" height="46" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="51.0" y="80.0" text-anchor="middle">Image</text>
<line x1="86" y1="75" x2="104" y2="75" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<rect x="106" y="46" width="104" height="58" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="158.0" y="66" text-anchor="middle">Backbone</text>
<text class="ar-s" x="158.0" y="82" text-anchor="middle">finds features</text>
<text class="ar-s" x="158.0" y="96" text-anchor="middle">ResNet or Swin</text>
<line x1="210" y1="75" x2="228" y2="75" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<rect x="230" y="46" width="118" height="58" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="289.0" y="66" text-anchor="middle">Pixel decoder</text>
<text class="ar-s" x="289.0" y="82" text-anchor="middle">3 feature sizes</text>
<text class="ar-s" x="289.0" y="96" text-anchor="middle">+ pixel embeddings</text>
<line x1="348" y1="61" x2="414" y2="61" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<text class="ar-s" x="381" y="57" text-anchor="middle">1/32</text>
<line x1="348" y1="75" x2="414" y2="75" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<text class="ar-s" x="381" y="71" text-anchor="middle">1/16</text>
<line x1="348" y1="89" x2="414" y2="89" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<text class="ar-s" x="381" y="85" text-anchor="middle">1/8</text>
<rect x="416" y="30" width="176" height="96" rx="5" style="fill: var(--viz-box); stroke: var(--color-text); stroke-width: 1.4"/>
<text class="ar-h" x="504.0" y="50" text-anchor="middle">Transformer decoder</text>
<text class="ar-s" x="504.0" y="66" text-anchor="middle">queries read the features</text>
<text class="ar-s" x="504.0" y="80" text-anchor="middle">9 layers: 1/32, 1/16, 1/8</text>
<text class="ar-s" x="504.0" y="94" text-anchor="middle">cycled three times</text>
<circle cx="470" cy="182" r="6" style="fill: var(--viz-gold)"/>
<circle cx="488" cy="182" r="6" style="fill: var(--viz-gold)"/>
<circle cx="506" cy="182" r="6" style="fill: var(--viz-gold)"/>
<text class="ar-g" x="488" y="206" text-anchor="middle">N learned queries</text>
<path d="M 488 174 L 488 130" style="stroke: var(--viz-gold); stroke-width: 1.4; fill: none" marker-end="url(#ar-ag)"/>
<rect x="614" y="30" width="92" height="42" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="660.0" y="50" text-anchor="middle">Class</text>
<text class="ar-s" x="660.0" y="66" text-anchor="middle">what it is</text>
<rect x="614" y="84" width="92" height="42" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="660.0" y="104" text-anchor="middle">Mask</text>
<text class="ar-s" x="660.0" y="120" text-anchor="middle">where it is</text>
<line x1="592" y1="56" x2="612" y2="51" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<line x1="592" y1="100" x2="612" y2="105" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<path d="M 660 126 C 660 160, 650 166, 630 166 L 600 166 C 580 166, 576 150, 576 130" style="stroke: var(--viz-gold); stroke-width: 1.4; stroke-dasharray: 4 3; fill: none" marker-end="url(#ar-ag)"/>
<text class="ar-g" x="614" y="188" text-anchor="middle">each mask limits</text>
<text class="ar-g" x="614" y="202" text-anchor="middle">where the next</text>
<text class="ar-g" x="614" y="216" text-anchor="middle">layer looks</text>
<path d="M 289 104 L 289 240 L 686 240 L 686 128" style="stroke: var(--viz-slate); stroke-width: 1.3; fill: none" marker-end="url(#ar-ar)"/>
<text class="ar-s" x="488" y="258" text-anchor="middle" style="fill: var(--viz-slate)">pixel embedding map: a mask is each query compared with every pixel here</text>
</svg>
<svg class="m2f-arch-v" viewBox="0 0 340 440" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mask2Former in one picture. The backbone and pixel decoder turn the image into features at three resolutions and a per-pixel embedding map. N learned queries read those features in a nine-layer Transformer decoder. Each query predicts a class and a mask, and each mask limits where the next layer looks.">
<defs><style>
.m2f-arch-v { display: none; }
@media (max-width: 600px) { .m2f-arch { display: none; } .m2f-arch-v { display: block; } }
</style><marker id="av-ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" style="fill: var(--viz-line)"/></marker><marker id="av-ag" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" style="fill: var(--viz-gold)"/></marker></defs>
<rect width="340" height="440" rx="6" style="fill: var(--color-bg)"/>
<rect x="75" y="10" width="100" height="34" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="125" y="32" text-anchor="middle">Image</text>
<line x1="125" y1="44" x2="125" y2="58" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<rect x="45" y="60" width="160" height="54" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="125" y="80" text-anchor="middle">Backbone</text>
<text class="ar-s" x="125" y="96" text-anchor="middle">finds features</text>
<text class="ar-s" x="125" y="109" text-anchor="middle">ResNet or Swin</text>
<line x1="125" y1="114" x2="125" y2="128" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<rect x="45" y="130" width="160" height="54" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="125" y="150" text-anchor="middle">Pixel decoder</text>
<text class="ar-s" x="125" y="166" text-anchor="middle">3 feature sizes</text>
<text class="ar-s" x="125" y="179" text-anchor="middle">+ pixel embeddings</text>
<line x1="90" y1="184" x2="90" y2="222" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<text class="ar-s" x="94" y="207">1/32</text>
<line x1="125" y1="184" x2="125" y2="222" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<text class="ar-s" x="129" y="207">1/16</text>
<line x1="160" y1="184" x2="160" y2="222" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<text class="ar-s" x="164" y="207">1/8</text>
<rect x="20" y="224" width="210" height="78" rx="5" style="fill: var(--viz-box); stroke: var(--color-text); stroke-width: 1.4"/>
<text class="ar-h" x="125" y="245" text-anchor="middle">Transformer decoder</text>
<text class="ar-s" x="125" y="262" text-anchor="middle">queries read the features</text>
<text class="ar-s" x="125" y="277" text-anchor="middle">9 layers: 1/32, 1/16, 1/8</text>
<text class="ar-s" x="125" y="292" text-anchor="middle">cycled three times</text>
<circle cx="270" cy="244" r="6" style="fill: var(--viz-gold)"/>
<circle cx="286" cy="244" r="6" style="fill: var(--viz-gold)"/>
<circle cx="302" cy="244" r="6" style="fill: var(--viz-gold)"/>
<path d="M 262 244 L 234 244" style="stroke: var(--viz-gold); stroke-width: 1.4; fill: none" marker-end="url(#av-ag)"/>
<text class="ar-g" x="286" y="266" text-anchor="middle">N learned queries</text>
<rect x="20" y="330" width="95" height="42" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="67.5" y="350" text-anchor="middle">Class</text>
<text class="ar-s" x="67.5" y="365" text-anchor="middle">what it is</text>
<rect x="135" y="330" width="95" height="42" rx="5" style="fill: var(--viz-box); stroke: var(--viz-line)"/>
<text class="ar-h" x="182.5" y="350" text-anchor="middle">Mask</text>
<text class="ar-s" x="182.5" y="365" text-anchor="middle">where it is</text>
<line x1="67.5" y1="302" x2="67.5" y2="328" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<line x1="182.5" y1="302" x2="182.5" y2="328" style="stroke: var(--viz-line); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<path d="M 230 351 L 246 351 Q 254 351 254 343 L 254 296 Q 254 288 246 288 L 234 288" style="stroke: var(--viz-gold); stroke-width: 1.4; stroke-dasharray: 4 3; fill: none" marker-end="url(#av-ag)"/>
<text class="ar-g" x="262" y="312">each mask</text>
<text class="ar-g" x="262" y="326">limits where</text>
<text class="ar-g" x="262" y="340">the next</text>
<text class="ar-g" x="262" y="354">layer looks</text>
<path d="M 45 157 L 10 157 L 10 392 L 182.5 392 L 182.5 375" style="stroke: var(--viz-slate); stroke-width: 1.3; fill: none" marker-end="url(#av-ar)"/>
<text class="ar-s" x="170" y="412" text-anchor="middle" style="fill: var(--viz-slate)">pixel embedding map: a mask is each</text>
<text class="ar-s" x="170" y="426" text-anchor="middle" style="fill: var(--viz-slate)">query compared with every pixel here</text>
</svg>
<figcaption>Overview. N learned queries read the image's features in nine decoder layers, cycling through three resolutions. Each query predicts a class and a mask, and each mask limits where the next layer looks.</figcaption>
</figure>

**How to read this.**

- **In five minutes:** the five steps and the picture above, then [Fig. 2](#fig-2), [Fig. 3](#fig-3), [Fig. 4](#fig-4), [Fig. 5](#fig-5) and [Fig. 6](#fig-6), which show the steps in order.
- **If you know DETR and the segmentation metrics:** start at §3. Mask2Former's own changes are §5 to §7, and §7 ends with what each one is worth.
- **If any of this is new:** start at §0, which builds every idea the rest uses. If attention and losses are familiar, start at §1.

Proofs and longer derivations are folded, and the main text never depends on them. In the figures, gold marks what the model predicts, green marks ground truth, and slate marks image features.

<details class="proof">
<summary>Contents</summary>

- **Background.** [0. The background you need](#0-the-background-you-need) · [1. Three ways to segment an image](#1-three-ways-to-segment-an-image) · [2. Mask classification](#2-mask-classification-predict-regions-not-pixels)
- **The model and its training.** [3. How a query becomes a mask](#3-how-a-query-becomes-a-mask) · [4. Training: one query per object](#4-training-one-query-per-object)
- **What Mask2Former changes.** [5. Masked attention](#5-masked-attention-each-query-reads-only-its-own-region) · [6. Making it affordable](#6-making-it-affordable) · [7. Smaller changes](#7-smaller-changes-and-what-each-part-is-worth)
- **Results.** [8. Results and limitations](#8-results-and-limitations) · [9. Conclusion](#9-conclusion) · [Appendix A](#appendix-a-implementation-notes) · [References](#references) · [Citation](#citation)

</details>

<details class="proof">
<summary>All symbols, for reference</summary>

| symbol | meaning |
|---|---|
| $I \in \mathbb{R}^{3\times H\times W}$ | input image. $H_l \times W_l$ is the size of the feature map used at decoder layer $l$ |
| $\Omega$ | the pixel grid, the set of pixel locations of the map under discussion. $\lvert\Omega\rvert$ is its size |
| $C$ | shared feature dimension of queries and image features (256 in the reference implementation) |
| $N$ | number of object queries (100 by default, 200 for the Swin-L panoptic and instance models) |
| $K$, $\varnothing$ | number of classes, and the "no-object" class appended to them. This $K$ is unrelated to §0's key matrix |
| $(m_i, c_i)$ | prediction $i$, with a soft mask $m_i \in [0,1]^{H\times W}$ and class $c_i \in \{1,\dots,K\}\cup\{\varnothing\}$ |
| $\hat p_i(c)$ | predicted probability that query $i$ has class $c$, from $\hat p_i = \operatorname{softmax}(W_{\text{cls}}\, q_i + b_{\text{cls}})$ with one output per class plus no-object |
| $q_i \in \mathbb{R}^C$ | the normalized feature of query $i$ that the class and mask heads read (§3) |
| $\pi$ | an assignment of predictions to targets, a permutation in $S_N$ (§4) |
| $\mathbf{X}_l \in \mathbb{R}^{N\times C}$ | query features after decoder layer $l$. $\mathbf{X}_0$ are the learnable input query features |
| $\widetilde{\mathbf{X}}_l$ | intermediate residual state immediately after adding the cross-attention output in layer $l$, before LayerNorm and the remaining sublayers |
| $\mathbf{Q}_l, \mathbf{K}_l, \mathbf{V}_l$ | attention projections. $\mathbf{Q}_l \in \mathbb{R}^{N\times C}$ comes from the queries, and $\mathbf{K}_l,\mathbf{V}_l \in \mathbb{R}^{H_lW_l\times C}$ come from image features |
| $Z_l$ | mask logits predicted at output $l$, one logit map per query |
| $B_l(i,x)$ | binary gate for query $i$ at location $x$, obtained by resizing $Z_l$, applying the sigmoid, and thresholding at $0.5$ |
| $\mathcal{M}_{l}(i,x)$ | additive attention mask built from $B_l(i,x)$. It is $0$ on allowed locations and $-\infty$ elsewhere |
| $\mathcal{E}_{\text{pixel}}$ | per-pixel embeddings from the pixel decoder, at stride 4 |
| $\sigma(\cdot)$ | the logistic sigmoid $\sigma(z) = 1/(1+e^{-z})$, which maps real scores into $(0,1)$ |
| $\mathbb{1}[\cdot]$ | the indicator, with $\mathbb{1}[P] = 1$ when condition $P$ holds and $0$ otherwise |
| $\lambda_{\text{ce}}, \lambda_{\text{dice}}, \lambda_{\text{cls}}$ | Binary cross-entropy (BCE), Dice, and class-loss weights. Their values are $5$, $5$, and $2$. The no-object class receives an additional $0.1$ multiplier |
| $K_{\text{pt}}$ | number of sampled points for mask losses, $12{,}544 = 112^2$ |

</details>

## 0. The background you need

Mask2Former is built from a few standard ideas, one per paragraph below. Skip any you already know.

**Feature maps.** A neural network called a backbone turns the image into a smaller grid, where each cell holds a vector that summarizes a patch of the image. That grid is a feature map, and its stride is the shrink factor. At stride 32, a 1024 by 1024 image becomes a 32 by 32 grid, so each cell stands for a 32 by 32 patch. Small objects lose detail at that scale, which is why Mask2Former also reads finer maps (§6).

**Embeddings and the dot product.** An embedding is a learned vector whose direction carries meaning. Related inputs tend to point in similar directions. The dot product measures how much two vectors agree: multiply them entry by entry, then add. For example, $(2, 1)\cdot(1, 2) = 4$, while $(2, 1)\cdot(-1, 2) = 0$. In general, for vectors $u$ and $v$ with angle $\theta$ between them,

$$
u^\top v = \sum_k u_k v_k = \|u\|\,\|v\|\cos\theta,
$$

where $u^\top$ is $u$ written as a row and $\|u\|$ is its length. The result is positive when the vectors point the same way, zero when they are perpendicular, and negative when they point apart. Mask2Former makes every mask this way (§3).

**Softmax.** Softmax turns a list of scores into weights that are positive and add up to 1. Take the scores 2.0, −0.6, −2.2 and 1.1. Exponentiating gives 7.39, 0.55, 0.11 and 3.00. Dividing each by their sum, 11.05, gives the weights 0.67, 0.05, 0.01 and 0.27. The largest score takes most of the weight, but no weight is exactly zero. In general, for scores $z_1, \dots, z_n$, which are also called logits,

$$
w_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

The only way to give an option a weight of exactly 0 is a score of $-\infty$, because $e^{-\infty} = 0$. The other weights still add up to 1. Masked attention in §5 is built on this.

**Attention.** Attention [[Vaswani et al. 2017](#ref-vaswani2017)] lets one vector, the query, collect information from many others. Each image location carries two vectors, both computed from its features by learned linear maps. Its key is compared with the query, and its value is what it hands over. The dot product of the query with each key gives a score, softmax turns the scores into weights, and the output is the values mixed by those weights. Fig. 0 runs this on four locations of a photo, with the scores of the softmax example.

<figure class="viz">
<video data-base="/assets/m2f/attention" width="1600" height="1000" muted playsinline preload="none" aria-label="Animation. A query is compared with four locations on a photo of a dog: 1 on its face, 2 and 3 on the blanket, 4 on its back. The scores 2.0, minus 0.6, minus 2.2 and 1.1 become exp values 7.39, 0.55, 0.11 and 3.00, which sum to 11.05. Dividing each by the sum gives the weights 0.67, 0.05, 0.01 and 0.27. The output is 0.67 v1 plus 0.05 v2 plus 0.01 v3 plus 0.27 v4."></video>
<figcaption>Fig. 0. Attention is a soft lookup: compare the query with each key, turn the scores into weights with softmax, then mix the values. The scores are made up for the example, and §5 reuses them.</figcaption>
</figure>

<details class="proof">
<summary>The matrix form, for many queries at once</summary>

For many queries at once, stack the $N$ queries as the rows of a matrix $Q$, and the keys and values of the $n$ locations as the rows of $K$ and $V$. Every row holds $d$ numbers. Then

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V,
$$

where each row of $QK^\top$ holds one query's scores, softmax turns each row into weights, and multiplying by $V$ mixes the values. Dividing by $\sqrt d$ keeps the scores from growing with the vector length $d$. In practice the model runs several lookups in parallel, called heads, each with its own learned projections.

</details>

**Transformer decoder layers.** A decoder layer updates every query's vector $x$ three times, each time adding the result to $x$, which is called a residual connection. Leaving out normalization and projections, Mask2Former's layer runs

$$
x \,\leftarrow\, x + \operatorname{Attention}(x,\ \text{image}), \qquad
x \,\leftarrow\, x + \operatorname{Attention}(x,\ x), \qquad
x \,\leftarrow\, x + \operatorname{FFN}(x).
$$

The first update is cross-attention, where the query reads the image. The second is self-attention, where the queries read each other, so they can coordinate, for example to avoid describing the same object twice. The third is a small feed-forward network (FFN) applied to each query on its own. Mask2Former stacks nine such layers.

**Losses and gradients.** A loss $\mathcal{L}$ is one number that measures how wrong the output is. Training nudges every weight against the gradient $\nabla_w \mathcal{L}$, the direction in weight space that raises the loss fastest. Mask2Former's loss is defined in §4.

## 1. Three ways to segment an image

Segmentation groups pixels and labels the groups. Datasets split categories into *things*, countable objects such as people and cars, and *stuff*, amorphous regions such as road and sky [[Kirillov et al. 2019](#ref-kirillov2019pan)]. Fig. 1 shows one street scene segmented all three ways.

<figure class="viz">
<svg class="m2f-tasks" style="min-width: 560px" viewBox="0 0 720 236" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="One street scene segmented three ways. Semantic segmentation merges two adjacent cars into one car region and labels sky and road. Instance segmentation keeps car 1 and car 2 apart and leaves sky and road unlabeled. Panoptic segmentation labels sky and road and keeps both cars apart.">
<defs><style>
.tk-lbl { font-family: var(--font-sans); fill: var(--color-text); font-size: 13px; }
.tk-sub { font-family: var(--font-sans); fill: var(--color-text-muted); font-size: 11px; }
.tk-in { font-family: var(--font-sans); fill: var(--color-text); font-size: 11px; }
</style></defs>
<rect width="720" height="236" rx="6" style="fill: var(--color-bg)"/>
<text class="tk-lbl" x="128.0" y="32" text-anchor="middle">semantic</text>
<rect x="24" y="46" width="208" height="84" style="fill: var(--viz-slate); fill-opacity: 0.14"/>
<rect x="24" y="130" width="208" height="44" style="fill: var(--viz-line); fill-opacity: 0.32"/>
<text class="tk-in" x="222" y="64" text-anchor="end">sky</text>
<text class="tk-in" x="222" y="164" text-anchor="end">road</text>
<rect x="24" y="46" width="208" height="128" style="fill: none; stroke: var(--viz-grid)"/>
<path style="fill: var(--viz-gold); fill-opacity: 0.55;" d="M 48 146 L 48 136 Q 48 132 52 132 L 62 132 L 70 120 L 98 120 L 108 132 L 120 132 Q 124 132 124 136 L 124 146 Z"/><circle style="fill: var(--viz-gold); fill-opacity: 0.55;" cx="64" cy="147" r="6"/><circle style="fill: var(--viz-gold); fill-opacity: 0.55;" cx="108" cy="147" r="6"/>
<path style="fill: var(--viz-gold); fill-opacity: 0.55;" d="M 124 146 L 124 136 Q 124 132 128 132 L 138 132 L 146 120 L 174 120 L 184 132 L 196 132 Q 200 132 200 136 L 200 146 Z"/><circle style="fill: var(--viz-gold); fill-opacity: 0.55;" cx="140" cy="147" r="6"/><circle style="fill: var(--viz-gold); fill-opacity: 0.55;" cx="184" cy="147" r="6"/>
<text class="tk-in" x="123" y="110" text-anchor="middle">car</text>
<text class="tk-sub" x="128.0" y="196" text-anchor="middle">one class per pixel:</text>
<text class="tk-sub" x="128.0" y="211" text-anchor="middle">the two cars merge</text>
<text class="tk-lbl" x="360.0" y="32" text-anchor="middle">instance</text>
<rect x="256" y="46" width="208" height="128" style="fill: none; stroke: var(--viz-grid)"/>
<path style="fill: var(--viz-gold); fill-opacity: 0.6; stroke: var(--color-text); stroke-width: 1.2;" d="M 280 146 L 280 136 Q 280 132 284 132 L 294 132 L 302 120 L 330 120 L 340 132 L 352 132 Q 356 132 356 136 L 356 146 Z"/><circle style="fill: var(--viz-gold); fill-opacity: 0.6; stroke: var(--color-text); stroke-width: 1.2;" cx="296" cy="147" r="6"/><circle style="fill: var(--viz-gold); fill-opacity: 0.6; stroke: var(--color-text); stroke-width: 1.2;" cx="340" cy="147" r="6"/>
<path style="fill: var(--viz-gold); fill-opacity: 0.25; stroke: var(--color-text); stroke-width: 1.2;" d="M 356 146 L 356 136 Q 356 132 360 132 L 370 132 L 378 120 L 406 120 L 416 132 L 428 132 Q 432 132 432 136 L 432 146 Z"/><circle style="fill: var(--viz-gold); fill-opacity: 0.25; stroke: var(--color-text); stroke-width: 1.2;" cx="372" cy="147" r="6"/><circle style="fill: var(--viz-gold); fill-opacity: 0.25; stroke: var(--color-text); stroke-width: 1.2;" cx="416" cy="147" r="6"/>
<text class="tk-in" x="318" y="110" text-anchor="middle">car 1</text>
<text class="tk-in" x="394" y="110" text-anchor="middle">car 2</text>
<text class="tk-sub" x="360.0" y="196" text-anchor="middle">one mask per thing:</text>
<text class="tk-sub" x="360.0" y="211" text-anchor="middle">sky and road left out</text>
<text class="tk-lbl" x="592.0" y="32" text-anchor="middle">panoptic</text>
<rect x="488" y="46" width="208" height="84" style="fill: var(--viz-slate); fill-opacity: 0.14"/>
<rect x="488" y="130" width="208" height="44" style="fill: var(--viz-line); fill-opacity: 0.32"/>
<text class="tk-in" x="686" y="64" text-anchor="end">sky</text>
<text class="tk-in" x="686" y="164" text-anchor="end">road</text>
<rect x="488" y="46" width="208" height="128" style="fill: none; stroke: var(--viz-grid)"/>
<path style="fill: var(--viz-gold); fill-opacity: 0.6; stroke: var(--color-text); stroke-width: 1.2;" d="M 512 146 L 512 136 Q 512 132 516 132 L 526 132 L 534 120 L 562 120 L 572 132 L 584 132 Q 588 132 588 136 L 588 146 Z"/><circle style="fill: var(--viz-gold); fill-opacity: 0.6; stroke: var(--color-text); stroke-width: 1.2;" cx="528" cy="147" r="6"/><circle style="fill: var(--viz-gold); fill-opacity: 0.6; stroke: var(--color-text); stroke-width: 1.2;" cx="572" cy="147" r="6"/>
<path style="fill: var(--viz-gold); fill-opacity: 0.25; stroke: var(--color-text); stroke-width: 1.2;" d="M 588 146 L 588 136 Q 588 132 592 132 L 602 132 L 610 120 L 638 120 L 648 132 L 660 132 Q 664 132 664 136 L 664 146 Z"/><circle style="fill: var(--viz-gold); fill-opacity: 0.25; stroke: var(--color-text); stroke-width: 1.2;" cx="604" cy="147" r="6"/><circle style="fill: var(--viz-gold); fill-opacity: 0.25; stroke: var(--color-text); stroke-width: 1.2;" cx="648" cy="147" r="6"/>
<text class="tk-in" x="550" y="110" text-anchor="middle">car 1</text>
<text class="tk-in" x="626" y="110" text-anchor="middle">car 2</text>
<text class="tk-sub" x="592.0" y="196" text-anchor="middle">every pixel labeled,</text>
<text class="tk-sub" x="592.0" y="211" text-anchor="middle">ids kept on things</text>
</svg>
<figcaption>Fig. 1. One street scene, three tasks.</figcaption>
</figure>

**Semantic segmentation** gives every pixel a class, so two touching cars become one car region. **Instance segmentation** returns one mask per thing, each with a confidence score, and ignores stuff. **Panoptic segmentation** [[Kirillov et al. 2019](#ref-kirillov2019pan)] does both: every pixel gets a class, and pixels of things also get an instance id.

**How the tasks are scored.** Each task has its own score, from 0 to 100, where higher is better: mIoU for semantic, AP for instance and PQ for panoptic segmentation. All three rest on intersection over union (IoU), the area two regions share divided by their combined area, which is 1 for identical regions and 0 for regions that do not touch.

The results in this article use two datasets: COCO [[Lin et al. 2014](#ref-lin2014)] for instance and panoptic segmentation, and ADE20K [[Zhou et al. 2017](#ref-zhou2017)] for semantic segmentation.

The exact definitions are folded below.

<details class="proof">
<summary>How mIoU is computed</summary>

Accumulate the predicted and ground-truth pixels of each class $c$ over the whole evaluation set, as pixel sets $P_c$ and $G_c$. Then

$$
\text{mIoU} = \frac{1}{|\mathcal{C}_{\mathrm{eval}}|}\sum_{c\in\mathcal{C}_{\mathrm{eval}}} \frac{|P_c \cap G_c|}{|P_c \cup G_c|},
\qquad
\mathcal{C}_{\mathrm{eval}} = \{c : |G_c| > 0\}
$$

[[Everingham et al. 2015](#ref-everingham2015)]. The average runs over the classes present in the ground truth, as in Detectron2's evaluator, which also rules out the $0/0$ case.

</details>

<details class="proof">
<summary>How COCO mask AP is computed</summary>

For each image and class, predictions are processed in descending score order. A prediction is a true positive when its best still-unclaimed ground truth of the same class in that image clears the current IoU threshold.

$$
\begin{gathered}
\text{IoU}(\bar m_i, g) = \frac{\sum_{x\in\Omega} \bar m_i(x)\, g(x)}{\sum_{x\in\Omega} \big(\bar m_i(x) + g(x) - \bar m_i(x)\, g(x)\big)},\\[4pt]
i \in \mathit{TP} \iff \max_{g \in \mathcal{G}_i} \text{IoU}(\bar m_i, g) \ge \tau,
\end{gathered}
$$

where the sums visit every pixel $x$ of the grid $\Omega$. COCO receives binary masks. Mask2Former's released inference code obtains $\bar m_i(x) = \mathbb{1}[m_i(x) > 0.5]$, which is $1$ on claimed pixels and $0$ elsewhere. The ground-truth mask $g(x)$ is also binary. On $0/1$ values a product acts as an AND, so the numerator counts the intersection. The combination $\bar m_i(x) + g(x) - \bar m_i(x)\, g(x)$ acts as an OR, so the denominator counts the union. Finally $\mathcal{G}_i$ holds the ground truths of the same class and image that remain unclaimed when prediction $i$ takes its turn. If $\mathcal{G}_i$ is empty the max is $-\infty$, an automatic false positive. Once a true positive claims a ground truth, later duplicates become false positives, which makes confidence ranking matter. Everything else is a false positive.

After matching within each image, COCO gathers the detections for each class across the dataset and sorts them by score. At any score cutoff, precision is the fraction of kept predictions that are true positives, and recall is the fraction of ground truths recovered. Sweeping the cutoff traces the precision-recall curve. COCO summarizes its interpolated precision at 101 recall levels and averages over ten IoU thresholds [[Lin et al. 2014](#ref-lin2014)]:

$$
P = \frac{|\mathit{TP}|}{|\mathit{TP}| + |\mathit{FP}|}, \qquad
R = \frac{|\mathit{TP}|}{|G|},
$$

$$
\text{AP} = \frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}
\frac{1}{101}\sum_{r\in\{0,0.01,\ldots,1\}} p_\tau^{\mathrm{interp}}(r).
$$

Here $|G|$ is the number of ground-truth masks, $\mathcal{T} = \{0.50, 0.55, \dots, 0.95\}$, and $p_\tau^{\mathrm{interp}}(r) = \max_{r' \ge r} p_\tau(r')$ is the interpolated precision, the best precision the curve reaches at recall $r$ or beyond. This is the per-category AP. COCO then averages over categories. Recall levels the curve never reaches count as zero precision. A category with no ground-truth instances in the evaluation set is excluded from the mean. This sketch omits crowd and ignore bookkeeping and the per-image detection cap.

</details>

<details class="proof">
<summary>How PQ is computed, and why its matches are unique</summary>

Evaluation first sets aside void pixels, which carry no label, and crowd regions, which group objects the annotators could not separate. Write $\mathit{TP}$ for matched pairs, $\mathit{FP}$ for unmatched predictions, and $\mathit{FN}$ for unmatched ground-truth segments. Then

$$
\text{PQ} = \frac{\sum_{(p,g)\in \mathit{TP}}\text{IoU}(p,g)}{|\mathit{TP}| + \tfrac12|\mathit{FP}| + \tfrac12|\mathit{FN}|}
=
\underbrace{\frac{\sum_{(p,g)\in \mathit{TP}}\text{IoU}(p,g)}{|\mathit{TP}|}}_{\text{SQ}}
\times
\underbrace{\frac{|\mathit{TP}|}{|\mathit{TP}| + \tfrac12|\mathit{FP}| + \tfrac12|\mathit{FN}|}}_{\text{RQ}}.
$$

A match requires $\text{IoU} > 0.5$. PQ is computed per class, then averaged over classes with at least one true-positive, false-positive, or false-negative segment. Segmentation quality (SQ) is the average IoU of the matched pairs. Recognition quality (RQ) is an F1-like term that penalizes missed and extra segments. The factorization follows by multiplying and dividing by $|\mathit{TP}|$ when at least one match exists. If $|\mathit{TP}| = 0$ but the denominator is nonzero, PQ is $0$ and the factorization is not used.

**Lemma (matches are unique).** *If predicted segments and ground-truth segments are each pairwise disjoint, as the panoptic format requires, then $\text{IoU} > 0.5$ pairs one-to-one. Each segment can match at most one segment from the other set.*

**Proof.** Suppose a prediction $p$ has $\text{IoU}(p, g) > \tfrac12$. Then

$$
\begin{aligned}
|p \cap g|
&> \tfrac12\,|p \cup g| && \text{definition of IoU, rearranged}\\[2pt]
&\ge \tfrac12\,|g| && \text{since } g \subseteq p \cup g.
\end{aligned}
$$

So every matching prediction covers more than half of $g$. Now suppose two predictions $p_1, p_2$ both matched $g$. The panoptic format keeps predictions disjoint, so $p_1 \cap g$ and $p_2 \cap g$ are disjoint subsets of $g$, each larger than $\tfrac12|g|$. Together they would hold more than $|g|$ pixels inside $g$, which is impossible. So at most one prediction matches $g$. The same argument runs the other way. Since the panoptic format also keeps the ground-truth segments disjoint, and $p \subseteq p \cup g$ gives $|p \cap g| > \tfrac12|p|$, each prediction matches at most one ground truth. $\blacksquare$

Above the $0.5$ threshold, matching is therefore unambiguous. Greedy matching, which pairs segments one at a time with the best still-unclaimed partner, is exact.

</details>

## 2. Mask classification: predict regions, not pixels

A model can produce a segmentation in two ways. It can label every pixel, or it can predict a set of regions and label each region. Mask2Former does the second.

**Labeling pixels.** A per-pixel classifier asks the same question at every pixel: which class is this? The fully convolutional network (FCN) [[Long et al. 2015](#ref-long2015)] made this the standard approach to semantic segmentation. It is trained with cross-entropy at every pixel, the negative log of the probability it gives the correct class.

The catch is object identity. A per-pixel classifier has no variable for which object a pixel belongs to, so it cannot separate two touching cars. Instance systems added extra machinery for that: region proposals, anchor boxes, grouping rules, or non-maximum suppression (NMS), a step that deletes overlapping duplicate detections.

**Predicting regions.** Mask classification predicts regions instead. The model keeps $N$ learned query vectors. Each query predicts one mask, which says where its segment is, and one class, which says what the segment is:

$$
\{(m_i, c_i)\}_{i=1}^{N}, \qquad m_i \in [0,1]^{H\times W},\quad c_i \in \{1,\dots,K\}\cup\{\varnothing\},
$$

where the mask $m_i$ gives every pixel a probability of belonging to segment $i$, $K$ is the number of classes, and $\varnothing$ is an extra "no object" class for queries that find nothing. The mask and the class come from the same query, so they describe the same segment. Object identity comes for free, because two cars are two queries. The design comes from DETR [[Carion et al. 2020](#ref-carion2020)], a detector built on learned queries, by way of MaskFormer [[Cheng et al. 2021](#ref-cheng2021)], Mask2Former's direct predecessor.

<details class="proof">
<summary>Where mask classification comes from</summary>

**Mask R-CNN** [[He et al. 2017](#ref-he2017)] predicts a binary mask inside each detected box. **DETR** [[Carion et al. 2020](#ref-carion2020)] replaced hand-built proposals with learned queries and one-to-one matching, and its panoptic extension decoded masks from those queries. **MaX-DeepLab** [[Wang et al. 2021](#ref-wang2021)] brought end-to-end set prediction to panoptic segmentation. **MaskFormer** [[Cheng et al. 2021](#ref-cheng2021)] showed that the same prediction format could outperform per-pixel models on semantic segmentation. **K-Net** [[Zhang et al. 2021](#ref-zhang2021)] pursued a related design with one dynamic convolution kernel per segment.

</details>

Fig. 2 shows one such query on a real photo.

<figure class="viz" id="fig-2">
<img src="/assets/m2f/mask_classification.webp" width="1572" height="568" loading="lazy" decoding="async" alt="A photograph of a French bulldog with its predicted mask in gold, and beside it the soft mask, gold where the probability is high, with its 0.5 level outlined.">
<figcaption>Fig. 2. One query predicts one region and its class. This is the dog query of the trained model, the released Swin-L checkpoint. Right, its mask as probabilities, white at 0 and gold at 1, with the 0.5 level in black. Left, the mask thresholded at 0.5. The query says dog with probability 0.999.</figcaption>
</figure>

The same format covers all three tasks. Semantic targets contain one mask per class present in the image, instance targets one mask per object, and panoptic targets one mask per thing or stuff segment.

Mask2Former's thesis is that this format was right, and that MaskFormer's decoder and training recipe held it back.

## 3. How a query becomes a mask

Mask2Former keeps MaskFormer's three parts, drawn in the [overview figure](#overview) at the top. Each part hands its output to the next.

1. A **backbone**, either a ResNet [[He et al. 2016](#ref-he2016)] or a Swin Transformer [[Liu et al. 2021](#ref-liu2021)], turns the image into feature maps.
2. A **pixel decoder** refines them into feature maps at three resolutions, 1/32, 1/16 and 1/8 of the input. It also produces a pixel embedding map $\mathcal{E}_{\text{pixel}}$ at 1/4 resolution, with one vector of $C = 256$ numbers per location.
3. A **Transformer decoder** updates the $N$ query vectors in nine layers, each reading one of the three feature maps (§5, §6).

After every decoder layer, two small heads read each query's vector $q_i$. The class head is a linear layer with a softmax, which gives one probability per class plus one for no-object. The mask head is a small multilayer perceptron (MLP) that turns the query into a mask embedding, a vector the same length as each pixel embedding. The mask is the sigmoid of their dot product at every pixel $x$:

$$
m_i(x) = \sigma\big(\operatorname{MLP}(q_i)^\top\, \mathcal{E}_{\text{pixel}}(x)\big),
$$

where $\sigma(z) = 1/(1+e^{-z})$ turns the dot product, called the mask logit, into a probability. A pixel whose embedding points the same way as the query's mask embedding gets a high probability. Every part, from the backbone to the heads, is trained together on the loss of §4, so the pixels of an object learn embeddings that line up with its query. Fig. 3 shows this on the real model.

<figure class="viz" id="fig-3">
<video data-base="/assets/m2f/mask-dot" width="1600" height="1000" muted playsinline preload="none" aria-label="Animation. Left, the dog photo with a grid of 260 sample cells. Each cell flies into a plot on the right as one dot. The horizontal axis is the direction of the dog query's mask embedding, and a dashed vertical line marks where the dot product is zero, where the sigmoid equals 0.5. The 59 dots right of the line turn gold, and so do their cells on the photo, which form a blocky outline of the dog. The formula reads mask = sigmoid(query dot pixel) greater than 0.5. The blocky mask then gives way to the full-resolution mask over all 60,800 cells."></video>
<figcaption>Fig. 3. A mask is one dot product per pixel. Each dot is one pixel of the trained model's embedding map. Across is how far the pixel's embedding points along the dog query's vector, and height only spreads the dots out. On the dashed line the dot product is 0, which the sigmoid turns into probability 0.5, and the pixels to its right form the mask. The animation places 260 sample pixels, drawn as cells on the photo, then applies the same rule to all 60,800.</figcaption>
</figure>

Masks are computed at 1/4 of the input resolution, 256 by 256 for a 1024 by 1024 image, and then upsampled. Because the heads run after every layer, each query makes ten predictions: one from the learnable initial queries and one after each of the nine layers. Training supervises all ten (§4.2), and each of the first nine also decides where the next layer may look (§5).

**From predictions to the output.** At inference, the final $N$ predictions become the answer each task needs. Panoptic output gives each pixel to the confident query with the highest class probability times mask probability. Instance output ranks every query and class pair by its probability and keeps the top 100 as detections. Semantic output weights each mask by its class probabilities and labels every pixel with the class that collects the most weight. The exact rules are folded below.

<details class="proof">
<summary>How the outputs are post-processed</summary>

Panoptic output first keeps queries whose best class is not no-object and whose class probability exceeds $0.8$. Each pixel chooses the kept query with the largest class-probability times mask-probability score. The pixel is written only if the winning query's mask probability is at least $0.5$. For each query, the code divides the number of pixels it wins by the number of pixels in its own thresholded mask and discards the segment when this ratio is below $0.8$. The final segment is the intersection of the winning pixels and that thresholded mask. Stuff segments with the same class are merged. Rejected pixels remain void.

Semantic output drops the no-object column and labels each pixel $x$ with $\arg\max_c \sum_i \hat p_i(c)\, m_i(x)$. Both of these rules come from MaskFormer.

Instance output flattens the $N\times K$ query-class probabilities and selects the highest-scoring pairs. The released code keeps the top 100 pairs per image. The same query mask can therefore appear with more than one class before the top-pair cutoff. For a selected pair $(i,c)$, the final score multiplies its class probability by the average mask probability over the predicted foreground.

$$
s_{i,c} = \hat p_i(c)\,
\frac{\sum_x m_i(x)\,\mathbb{1}[m_i(x)>0.5]}
{\sum_x \mathbb{1}[m_i(x)>0.5] + \varepsilon}.
$$

The implementation uses $\varepsilon = 10^{-6}$, so an empty foreground scores zero. This mask-weighted score is Mask2Former's addition.

</details>

## 4. Training: one query per object

Training has to decide which query should predict which object, and nothing in the output says so. The $N$ predictions form a set. Query 7 is not meant to predict object 7, and shuffling the queries only reorders the same predictions. So the loss works in two steps: matching pairs each real object with one query, and the loss then scores each pair. Mask2Former inherits this from DETR and MaskFormer. If you know DETR's matching, skip to the last paragraph of §4.2, on deep supervision.

### 4.1 Matching predictions to objects

Letting each prediction pick the target it fits best does not work: two queries can pick the same object and leave another with none. Matching instead gives each target exactly one prediction, choosing the pairing with the lowest total cost. Every prediction left over learns to say no-object. That includes a second query on an object that is already taken, so the model learns not to predict duplicates, and inference needs no non-maximum suppression.

Try a small case first. Fig. 4 holds the costs of pairing three predictions with three targets. Before it plays, find the cheapest way to give each target its own prediction.

<figure class="viz" id="fig-4">
<video data-base="/assets/m2f/matching" width="1600" height="1000" muted playsinline preload="none" aria-label="Animation. A 3 by 3 cost matrix with predictions as rows and targets as columns. Row 1 reads 4, 1, 3. Row 2 reads 2, 0, 5. Row 3 reads 3, 2, 2. Cheapest first picks 0 for prediction 2 and target 2, then 2 for prediction 3 and target 3, and is left with 4 for prediction 1 and target 1, a total of 6. A second copy shows the best assignment: 1 for prediction 1 and target 2, 2 for prediction 2 and target 1, and 2 for prediction 3 and target 3, a total of 5."></video>
<figcaption>Fig. 4. Matching is a global choice. Each entry is the cost of giving a target (column) to a prediction (row), lower when the prediction's class and mask fit the target better. Taking the cheapest pair first locks in the 0 and the 2 and is left with a 4, for a total of 6. The best assignment accepts one slightly worse pair and pays 5 in total. The Hungarian algorithm finds it.</figcaption>
</figure>

**The cost.** In the model, the cost of pairing prediction $i$ with target $j$ has three parts. A class term rewards a prediction that already gives the target's class $c^{\text{gt}}_j$ a high probability. Two mask terms measure how far the predicted mask $m_i$ is from the target mask $m^{\text{gt}}_j$: binary cross-entropy (BCE), which scores each pixel on its own, and Dice, which scores the overlap of the whole region (both in §4.2). In full,

$$
\text{cost}(i,j) = \mathbb{1}\!\left[c^{\text{gt}}_j \ne \varnothing\right]\Big(\!-\lambda_{\text{cls}}\,\hat p_{i}(c^{\text{gt}}_j) + \lambda_{\text{ce}}\,\mathcal{L}_{\text{ce}}\big(m_{i}, m^{\text{gt}}_j\big) + \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}}\big(m_{i}, m^{\text{gt}}_j\big)\Big),
$$

where the indicator $\mathbb{1}[\cdot]$ is 1 when its condition holds and 0 otherwise, so pairing a query with no object costs nothing. The weights $\lambda_{\text{cls}}$, $\lambda_{\text{ce}}$ and $\lambda_{\text{dice}}$ are 2, 5 and 5, and the minus sign makes a confident correct class lower the cost.

**The assignment.** With 100 queries there are far too many pairings to try them all. The Hungarian algorithm [[Kuhn 1955](#ref-kuhn1955), [Munkres 1957](#ref-munkres1957)] finds the cheapest one directly, in $O(N^3)$ time in its refined form. In code it is one call to `scipy.optimize.linear_sum_assignment`. The two folds below show why its answer is optimal and run it by hand on the matrix of Fig. 4.

<details class="proof">
<summary>The dual certificate for optimality</summary>

Assume an image holds at most $N$ objects, and fill the target list up to $N$ with no-object entries $\varnothing$. Then every query gets exactly one target. An assignment is a permutation $\pi \in S_N$, where $S_N$ is the set of all $N!$ reorderings and $\pi(j)$ is the prediction given to target $j$. Its total cost is

$$
J(\pi) = \sum_{j=1}^{N} \text{cost}(\pi(j),\, j),
$$

and the matcher picks $\hat\pi = \arg\min_{\pi} J(\pi)$.

The assignment minimization has a companion maximization, [its dual](https://en.wikipedia.org/wiki/Dual_linear_program), whose value never exceeds the original minimum. The dual attaches a potential to each row, $u_i$, and to each column, $v_j$,

$$
\max_{u,v\in\mathbb{R}^N}\ \sum_i u_i + \sum_j v_j
\quad\text{s.t.}\quad
u_i + v_j \le \text{cost}(i,j)\ (\forall i,j),
$$

and dual feasibility is exactly the reduced cost $\tilde c(i,j) = \text{cost}(i,j) - u_i - v_j$ staying nonnegative everywhere.

**The optimality certificate.** Take any row potentials $u_1,\dots,u_N$ and column potentials $v_1,\dots,v_N$, and sum the reduced costs along an assignment $\pi$:

$$
\begin{aligned}
\sum_{j=1}^{N} \tilde c(\pi(j),\, j)
&= \sum_{j=1}^{N}\big[\text{cost}(\pi(j),\, j) - u_{\pi(j)} - v_j\big] && \text{definition of } \tilde c\\[2pt]
&= \sum_{j=1}^{N}\text{cost}(\pi(j),\, j) - \sum_{j=1}^{N} u_{\pi(j)} - \sum_{j=1}^{N} v_j && \text{split the sum}\\[2pt]
&= J(\pi) - \sum_{i=1}^{N} u_i - \sum_{j=1}^{N} v_j && \text{reindex the } u \text{ sum}.
\end{aligned}
$$

The reindexing is legitimate because $\pi$ is a permutation. The map $j \mapsto \pi(j)$ hits every row index once, so $\sum_j u_{\pi(j)} = \sum_i u_i$. The two potential sums do not depend on $\pi$, so the potentials shift every assignment's total by the same constant, and $\arg\min_\pi$ is unchanged. Call the potentials feasible when $\tilde c(i,j) \ge 0$ everywhere, and an edge tight when $\tilde c(i,j) = 0$.

**Lemma (a tight assignment is optimal).** *If feasible potentials admit an assignment $\pi$ that uses only tight edges, then $\pi$ minimizes $J$.*

**Proof.** Let $\pi'$ be any assignment. Rearranging the certificate identity,

$$
\begin{aligned}
J(\pi')
&= \sum_{j=1}^{N} \tilde c(\pi'(j),\, j) + \sum_{i=1}^{N} u_i + \sum_{j=1}^{N} v_j && \text{the identity above}\\[2pt]
&\ge \sum_{i=1}^{N} u_i + \sum_{j=1}^{N} v_j && \text{since every } \tilde c \ge 0.
\end{aligned}
$$

So $\sum_i u_i + \sum_j v_j$ lower-bounds the cost of every assignment. The tight $\pi$ uses only edges with $\tilde c(\pi(j),\, j) = 0$, so its first sum vanishes and $J(\pi) = \sum_i u_i + \sum_j v_j$. That equals the lower bound, so no assignment costs less. $\blacksquare$

This equality is [complementary slackness](https://en.wikipedia.org/wiki/Linear_programming#Complementary_slackness). With respect to an optimal dual solution, an optimal assignment uses only edges with $\tilde c=0$.

</details>

<details class="proof">
<summary>A worked 3&#215;3 Hungarian run and the O(N³) bound</summary>

**The primal-dual algorithm.** Instead of checking all $N!$ assignments, the Hungarian method grows a matching while keeping matched edges tight and every reduced cost nonnegative. Consider three predictions against three targets, with prediction $i$ in row $i$, target $j$ in column $j$, and the costs collected in a matrix written $C$ for this example only.

$$
C = \begin{pmatrix} 4 & 1 & 3\\ 2 & 0 & 5\\ 3 & 2 & 2 \end{pmatrix}.
$$

**Make zeros.** Start with $u = v = 0$. Row and column reductions create enough zero reduced costs to begin matching. Subtract each row's minimum from its row, which sets $u_i = \min_j \text{cost}(i,j)$, then subtract each remaining column minimum, $v_j = \min_i \tilde c(i,j)$. The certificate identity makes this legal because the potentials shift every assignment by the same constant. Here $u = (1, 0, 2)$ and then $v = (1, 0, 0)$.

$$
\tilde c = C - u - v = \begin{pmatrix} 2 & 0 & 2\\ 1 & 0 & 5\\ 0 & 0 & 0 \end{pmatrix}.
$$

Every row and every column now holds a zero. The column pass cannot remove a row's zero because that zero sits in a column whose minimum is already $0$, so $v_j=0$ there and the entry stays unchanged.

**Match on zeros.** Pair rows to columns through tight edges only, never reusing a row or a column. Row 1 takes column 2 and row 3 takes column 1. Row 2's only zero sits in column 2, already taken, so the matching stalls at two pairs.

The tool for getting unstuck is the augmenting path. Call a row or column free while it has no partner. An augmenting path starts at a free row and steps to a column through a zero. From that column it follows the matched edge to the partner row, then steps to a new column through another zero. It keeps alternating until it lands on a free column. Flip every edge along it, so matched edges become unmatched and unmatched edges become matched. Each interior row and column trades one partner for another while the two free endpoints gain one, so the matching grows by exactly one pair. Matching as far as possible means flipping augmenting paths until none exists.

None exists here. From the free row 2 the only zero leads to column 2, column 2's partner is row 1, and row 1 owns no other zero to continue through.

**Create a new zero.** The rows and columns just visited form the alternating tree, here rows $I = \{2, 1\}$ and columns $J = \{2\}$. These names are local to this example. Progress needs a zero on an edge from a row in $I$ to a column outside $J$. Take the smallest remaining reduced cost on such an edge.

$$
\delta = \min_{i\in I,\ j\notin J}\tilde c(i,j) = \tilde c(2,1) = 1,
$$

and pay for it with the potentials. Raise $u_i$ by $\delta$ on every tree row and lower $v_j$ by $\delta$ on every tree column. Call this move a lift. Three checks, one per kind of entry, show it is safe.

- *Tree row, outside column.* The reduced cost drops by $\delta$. By the choice of $\delta$ nothing goes negative, and the minimizer lands exactly on zero. That is the purchased edge, $(2,1)$ here.
- *Tree row, tree column.* The raise and the drop cancel, so every edge inside the tree, the matched ones included, stays exactly as tight as it was.
- *Outside row, tree column.* The reduced cost rises by $\delta$, harmless for nonnegativity, and the only zeros this can erase are unmatched ones, because a tree column's partner sits inside the tree.

So the invariants hold and the tree gains ground. The dual objective $\sum_i u_i + \sum_j v_j$ also rises by $\delta$, because the tree always holds one more row than column. Every column it reaches is matched, since a free column would already end an augmenting path. Each matched column brings its partner row along, so only the root row lacks a column. The lower bound climbs toward the optimum with every lift. With $u = (2, 1, 2)$ and $v = (1, -1, 0)$,

$$
\tilde c = \begin{pmatrix} 1 & 0 & 1\\ 0 & 0 & 4\\ 0 & 1 & 0 \end{pmatrix}.
$$

**Augment.** The purchased zero at $(2,1)$ extends the search. Row 2 now reaches column 1, column 1's partner row 3 joins, and row 3 owns a zero in column 3, which is free. That is an augmenting path, row 2 to column 1 to row 3 to column 3. Flip it. Row 2 takes column 1, row 3 moves to column 3, row 1 keeps column 2, and the matching is perfect.

**Read off the certificate.** In the original matrix the assignment costs $C_{12} + C_{21} + C_{33} = 1 + 2 + 2 = 5$, and the potentials sum to $(2 + 1 + 2) + (1 - 1 + 0) = 5$. Primal equals dual, so by the lemma no assignment can cost less. The potentials are an optimal dual solution, and the equality that just closed is complementary slackness checked by hand. In general, match through zeros until stuck, buy the cheapest escape zero, augment the moment a free column comes into reach, and stop when every row has a partner.

Each augmentation adds one edge, so at most $N$ augmentations finish the matching. One piece of bookkeeping makes each augmentation cost $O(N^2)$. For every column outside the tree, keep its current slack $\min_{i\in I}\tilde c(i,j)$ and update it in $O(N)$ when a row joins. Each $\delta$ is then read from the slacks rather than found by rescanning the matrix. Every lift either reaches a free column and enables an augmentation, or adds a matched column and its partner row to the tree. Thus at most $N$ lifts occur between augmentations, giving $O(N^3)$ work overall. The Jonker-Volgenant routine called by `scipy` is a faster-constant refinement of the same primal-dual idea.

</details>

### 4.2 The training loss

Each output is trained with three terms: a class loss on all $N$ queries, and two mask losses on the queries matched to real objects. The loss weights them as

$$
\mathcal{L} = 2\,\mathcal{L}_{\text{cls}} + 5\,\mathcal{L}_{\text{ce}} + 5\,\mathcal{L}_{\text{dice}}.
$$

The class loss $\mathcal{L}_{\text{cls}}$ is cross-entropy, with no-object targets weighted 0.1, because most queries are unmatched: with 100 queries and 5 objects, 95 learn no-object. BCE, $\mathcal{L}_{\text{ce}}$, scores each mask point on its own and keeps pushing a point that is confidently wrong. Dice, $\mathcal{L}_{\text{dice}}$ [[Milletari et al. 2016](#ref-milletari2016)], scores the overlap of the whole region, from 0 for a perfect match to almost 1 for none, so a small object counts as much as a large one at the same fraction of overlap. Its push on a confidently wrong point fades, which is why BCE is kept beside it. Both are evaluated at sampled points (§6.2).

<details class="proof">
<summary>The loss formulas, their gradients, and what BCE adds to Dice</summary>

**Class loss.** After matching, write $y_i$ for the class that query $i$ should predict: its target's class if it was matched, and $\varnothing$ if not. For one image, the class loss is a weighted cross-entropy over the query slots,

$$
\mathcal{L}_{\text{cls}}
= \frac{\sum_{i=1}^{N} w(y_i)\big[-\log \hat p_i(y_i)\big]}
{\sum_{i=1}^{N} w(y_i)},
\qquad
w(y_i) =
\begin{cases}
1 & y_i \ne \varnothing,\\
0.1 & y_i = \varnothing.
\end{cases}
$$

Mask2Former inherits the $0.1$ weight from DETR. The paper's text states it as $\lambda_{\text{cls}} = 2.0$ for matched predictions and $0.1$ for no-object. The released code applies the $0.1$ inside this weighted mean, which is then scaled by $2$.

**BCE.** At a point $x$ with mask logit $z_x$, prediction $m_x = \sigma(z_x)$, and target $g_x \in \{0,1\}$, it is

$$
\ell_{\text{ce}}(x) = -\big[g_x \log \sigma(z_x) + (1-g_x)\log\big(1-\sigma(z_x)\big)\big].
$$

Its gradient with respect to the logit is simply

$$
\frac{\partial \ell_{\text{ce}}}{\partial z_x} = \sigma(z_x) - g_x,
$$

so every point is pushed toward its target, and a confidently wrong point gets close to the largest push, of size 1.

**Dice.** Write $O = \sum_x m_xg_x$ for the overlap and $S = \sum_xm_x + \sum_xg_x$ for the size sum. The released implementation computes

$$
\mathcal{L}_{\text{dice}}(m,g) = 1 - \frac{2O+1}{S+1},
$$

where the added 1s keep the ratio defined when both masks are empty.

**BCE in the released code.** The same gradient holds for bilinearly sampled targets $g_x \in [0,1]$. The released loss averages BCE over sampled points and normalizes by the number of matched masks. BCE has no mask-level normalization for foreground area. Under dense evaluation, or uniform sampling in expectation, a large foreground contributes more positive terms than a small one at comparable pointwise errors. The training sampler in §6.2 can change those proportions by favoring logits near zero.

**Dice's gradient.** Milletari's original Dice uses sums of squares in the denominator. Dice couples all sampled points through one region-level score. Write the smoothing constant as $\varepsilon = 1$. The gradient at a single point is

$$
\frac{\partial \mathcal{L}_{\text{dice}}}{\partial m_x}
= -\,\frac{2\,g_x\,(S+\varepsilon) - (2O+\varepsilon)}{(S+\varepsilon)^2}.
$$

To derive it, differentiate with respect to the single probability $m_x$. Every other term of each sum is constant, so

$$
\frac{\partial O}{\partial m_x} = g_x, \qquad \frac{\partial S}{\partial m_x} = 1.
$$

Then, using the quotient rule in the form $(u/v)' = (u'v - uv')/v^2$,

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{\text{dice}}}{\partial m_x}
&= -\,\frac{\partial}{\partial m_x}\!\left(\frac{2O+\varepsilon}{S+\varepsilon}\right) && \text{drop the constant } 1\\[4pt]
&= -\,\frac{2\,(\partial O/\partial m_x)\,(S+\varepsilon) - (2O+\varepsilon)\,(\partial S/\partial m_x)}{(S+\varepsilon)^2} && \text{quotient rule}\\[4pt]
&= -\,\frac{2\,g_x\,(S+\varepsilon) - (2O+\varepsilon)}{(S+\varepsilon)^2} && \text{insert the partials}.
\end{aligned}
$$

**How Dice scales with area.** When $S>0$, the unsmoothed form gives an exact view of area scaling. Tile $k$ disjoint copies of the same prediction and target pattern. Then $O$ becomes $kO$ and $S$ becomes $kS$, so the Dice value is unchanged. In the derivative above with $\varepsilon=0$, the numerator grows by a factor of $k$ while the denominator grows by a factor of $k^2$. Each copied point therefore receives $1/k$ of the original derivative. If both masks are empty, the unsmoothed ratio is undefined, which is one reason the implementation adds $1$. This area normalization does not make the BCE component scale-invariant.

**What BCE adds to Dice.** Dice has a weakness one step further back, in logit space. When the prediction confidently misses the target, $m_x$ is near 0 where $g_x = 1$. The chain rule then multiplies $\partial\mathcal{L}_{\text{dice}}/\partial m_x$ by $\sigma'(z_x) \approx 0$, so the gradient vanishes through the saturated sigmoid. BCE cancels that factor exactly, as its logit gradient $\sigma(z_x) - g_x$ shows. BCE therefore supplies a direct logit gradient at every point, while Dice adds region-normalized overlap.

</details>

<details class="proof">
<summary>Why not focal loss, as in MaskFormer?</summary>

MaskFormer used focal loss [[Lin et al. 2017](#ref-lin2017)] with weight $20$. Focal loss down-weights already-confident predictions, so easy background points contribute less. Mask2Former uses plain BCE at weight $5$. One plausible reason is that the training sampler of §6.2 spends most of its budget on uncertain locations rather than easy background, reducing the imbalance that focal loss was designed to address. The paper does not isolate this change.

</details>

**Deep supervision.** All ten outputs of §3 are trained with this loss, through the same shared heads. The matching is recomputed for each output, so every intermediate prediction trains toward its own assignment. This matters more than usual here: the intermediate masks become the attention masks of §5, so these losses also steer where each layer looks.

## 5. Masked attention: each query reads only its own region

In a standard decoder layer, every query reads the whole image through cross-attention. A query meant to describe the dog still spends attention on every patch of blanket. Global attention can learn where an object is, but the authors argue that learning this is slow, as earlier DETR studies had found [[Gao et al. 2021](#ref-gao2021), [Sun et al. 2021](#ref-sun2021)]. Masked attention removes the problem by construction: each query reads only inside the mask it predicted at the previous layer.

### 5.1 Why global attention spreads thin

A small calculation shows how the background wins. Suppose the object covers 2 percent of the image, and every object location scores 2 higher than every background location. There are 49 background locations for each object location, and each of them gets $e^{-2} \approx 0.135$ of an object location's weight. The object's share of the attention is then only

$$
\frac{1}{1 + 49\,e^{-2}} \approx \frac{1}{1 + 49 \cdot 0.135} = \frac{1}{7.615} \approx 0.13.
$$

The background wins by numbers, not by score. Real scores vary, so this is only a toy, but it shows why reading only the object's region can make learning easier.

### 5.2 The masked update

Masked attention keeps the attention step of Fig. 0 and adds one term to the scores. Before reading on, take the four locations of Fig. 0 and suppose the previous mask keeps locations 1 and 4. What should happen to the weights of locations 2 and 3, and to the other two? Fig. 5 plays it out.

<figure class="viz" id="fig-5">
<video data-base="/assets/m2f/masked-attention" width="1600" height="1000" muted playsinline preload="none" aria-label="Animation. The four locations of Fig. 0 with the model's real dog mask drawn over the photo. Locations 1 and 4 are inside the mask, 2 and 3 outside. The blocked scores become minus infinity, so their exp values and weights become 0. The sum drops from 11.05 to 10.39, and the kept weights become 0.71 and 0.29. The output is 0.71 v1 plus 0.29 v4."></video>
<figcaption>Fig. 5. Masked attention zeroes the weights outside the previous mask and rescales the rest. A location's score says how well it matches the query, its weight is exp(score) divided by the sum over locations, and the output mixes the locations' vectors v by weight. The scores are made up. The mask is the model's real dog mask, which keeps locations 1 and 4. The other two get score −∞ and weight 0, so the kept weights grow to 0.71 and 0.29.</figcaption>
</figure>

The gain is small here because locations 2 and 3 held little weight. In the toy image of §5.1, where the background held 87 percent of the attention, a gate that keeps only the object lifts the object's share from 0.13 to 1.

In symbols, masked attention adds a mask $\mathcal{M}$ to the scores before the softmax. For query $i$, with the softmax taken over image locations $x$,

$$
\text{weight}_i(x) = \operatorname{softmax}_x\!\big(\text{score}_i(x) + \mathcal{M}_i(x)\big),
\qquad
\mathcal{M}_i(x) =
\begin{cases}
0 & \text{inside query } i\text{'s previous mask},\\
-\infty & \text{outside it}.
\end{cases}
$$

The previous mask is the one the query predicted at the layer before, resized to the feature map this layer reads and thresholded at probability 0.5. The first layer uses the mask predicted from the learned starting queries (§7). Self-attention, the next step in the layer, still carries information between queries, so a query that reads only its own region can still learn from the others.

<details class="proof">
<summary>The paper's equations</summary>

The paper writes the standard cross-attention step of layer $l$ as

$$
\widetilde{\mathbf{X}}_l = \operatorname{softmax}\!\big(\mathbf{Q}_l \mathbf{K}_l^{\top}\big)\,\mathbf{V}_l + \mathbf{X}_{l-1},
\tag{1}
$$

where $\mathbf{Q}_l$ comes from the queries after the previous layer, $\mathbf{X}_{l-1}$, and $\mathbf{K}_l$ and $\mathbf{V}_l$ come from the image features that layer $l$ reads. $\widetilde{\mathbf{X}}_l$ is the queries' state right after this step. Self-attention and the feed-forward network then turn it into the layer's output $\mathbf{X}_l$. The equation leaves out LayerNorm, the $1/\sqrt d$ scaling, the heads and the positional embeddings of §7.

Masked attention adds the mask $\mathcal{M}_{l-1}$ to the scores before the softmax:

$$
\widetilde{\mathbf{X}}_l = \operatorname{softmax}\!\big(\mathcal{M}_{l-1} + \mathbf{Q}_l \mathbf{K}_l^{\top}\big)\,\mathbf{V}_l + \mathbf{X}_{l-1},
\tag{2}
$$

$$
\begin{aligned}
B_{l-1}(i,x)
&= \mathbb{1}\!\left[\sigma\!\left(\operatorname{resize}(Z_{l-1})_i(x)\right) \ge 0.5\right],\\[4pt]
\mathcal{M}_{l-1}(i,x)
&=
\begin{cases}
0 & \text{if } B_{l-1}(i,x) = 1,\\[2pt]
-\infty & \text{otherwise,}
\end{cases}
\end{aligned}
\tag{3}
$$

where $Z_{l-1}$ holds the previous output's mask logits, one map per query. Each query's logits are bilinearly resized to the resolution of the map that layer $l$ reads, then passed through the sigmoid $\sigma$ and thresholded at $0.5$. The result is the gate $B_{l-1}$, which is 1 where the query may look. The first layer uses $Z_0$, predicted directly from the learnable queries (§7).

</details>

Does it help? The paper swaps out one part at a time to find out. With a ResNet-50 (R50) backbone on COCO, plain cross-attention reaches 37.8 AP and masked attention 43.7. Even mask pooling, which simply averages the features inside the mask, reaches 43.1. So on COCO, most of the gain comes from limiting where a query reads, and masked attention adds a learned weighting inside the region.

The gate is a hard threshold, so no gradient passes through it. The intermediate masks learn from deep supervision instead (§4.2), which compares each intermediate prediction directly with the target mask.

<details class="proof">
<summary>What happens when a mask is empty</summary>

If a query's thresholded mask is empty at some scale, every score in its row would be $-\infty$ and the softmax would be undefined. The code then lets that query read every location for one layer.

```python
attn_mask = (mask_logits.sigmoid() < 0.5).flatten(2)   # True = forbidden
attn_mask[attn_mask.all(dim=-1)] = False               # Fall back to global attention.
```

Without the guard, the row's softmax denominator is zero, which can introduce a NaN. The following self-attention mixes information across queries, so that invalid value can contaminate other query states, the loss, and the gradients. Flipping the row to fully unmasked gives the softmax every location to normalize over.

</details>

### 5.3 The effect, measured and seen

The paper also measures how much of each query's attention lands inside its own object, the true segment it was matched to. With standard cross-attention the share is about 0.20. With masked attention it is 0.59. It stays below 1 partly because the gate is the previous predicted mask, at a coarse resolution, not the true one.

Fig. 6 shows the mechanism in the released Swin-L checkpoint. The query's class settles early: it says dog with probability 0.98 after layer 2, and at least 0.998 from layer 3 on.

<figure class="viz" id="fig-6">
<video data-base="/assets/m2f/layers" width="1600" height="1000" muted playsinline preload="none" aria-label="Animation. A three by three grid of the dog photo fills in, one cell per decoder layer, from layer 1 to layer 9. Columns are the 1/32, 1/16 and 1/8 feature maps, so the cells get finer from left to right. In each cell, gold marks where the dog query's attention goes, and an outline marks the gate from the previous mask. No attention falls outside the outline."></video>
<figcaption>Fig. 6. Each layer reads only inside the previous layer's mask. Each cell shows where the dog query reads in one of the nine decoder layers: rows are layers 1 to 3, 4 to 6 and 7 to 9, and columns the 1/32, 1/16 and 1/8 maps. The outline is the previous layer's mask. Dark gold is the smallest area that holds half of the query's attention, and light gold 90 percent.</figcaption>
</figure>

## 6. Making it affordable

Masked attention is the main idea. Two more ideas make it affordable: each layer reads only one feature resolution, and the mask losses are computed on sampled points instead of every pixel.

### 6.1 One resolution per layer

Small objects need fine feature maps, but attention over a fine map is expensive. Its cost grows with the number of locations. For a 1024 by 1024 input, the 1/32, 1/16 and 1/8 maps hold 1,024, 4,096 and 16,384 locations, so the finest map costs about sixteen times more than the coarsest.

Mask2Former reads one scale per layer. Layers 1, 2 and 3 read the 1/32, 1/16 and 1/8 maps, and the cycle repeats twice more, as Fig. 6 shows on the dog query. This round-robin reaches 43.7 AP at 226 GFLOPs, billions of operations per image. Feeding all three scales to every layer costs 21 more GFLOPs and is no better overall: 0.3 AP higher, the same PQ, and 0.9 mIoU lower.

The pixel decoder builds these maps by letting every location mix in information from all three scales. It uses multi-scale deformable attention (MSDeformAttn) from Deformable DETR [[Zhu et al. 2021](#ref-zhu2021)], in which each location reads a few learned sample points per scale rather than every location.

### 6.2 Mask losses on 12,544 points

For every output it trains, MaskFormer compared whole masks pixel by pixel twice: once for every prediction-target pair to build the matching costs, and again for the matched pairs to compute the loss. That limited training to one image per 32 GB GPU. Following PointRend [[Kirillov et al. 2020](#ref-kirillov2020)], Mask2Former evaluates these terms at only $K_{\text{pt}} = 12{,}544 = 112^2$ sampled points per mask.

**Matching uses shared uniform points.** One set of random points, spread uniformly, is shared by every prediction and target mask in the image, which makes the cost matrix cheaper to build.

**Training uses uncertain points.** For each matched mask, the PointRend sampler draws $3K_{\text{pt}} = 37{,}632$ uniform candidates [[Kirillov et al. 2020](#ref-kirillov2020)]. It keeps the 9,408 where the prediction is least sure, the logits closest to zero, then adds 3,136 fresh uniform points. The uncertainty score depends on the prediction, not the ground truth. It is also relative: the sampler always keeps the least-sure quarter of its candidates, so where the unsure band is thin, it also takes points that are only somewhat unsure. Fig. 7 runs both rules on the dog prediction.

<figure class="viz" id="fig-7">
<video data-base="/assets/m2f/point-sampling" width="1600" height="1000" muted playsinline preload="none" aria-label="Animation. Two copies of the dog photo with the predicted mask outlined in gold. Left, matching: 400 uniform points appear, one set shared by every mask. Right, training: 1,200 uniform candidates appear. The 300 whose logits are closest to zero turn gold and the other 900 disappear, leaving gold points along the predicted outline and around the gift box. Then 100 fresh uniform points are added."></video>
<figcaption>Fig. 7. Matching spreads its points evenly, and training puts most of its points where the prediction is unsure. Both rules run on the dog prediction of Fig. 2, with 400 points per mask instead of 12,544 so they stay visible. Left, matching: 400 uniform points, one set shared by every mask. Right, training: of 1,200 uniform candidates, the 300 with logits closest to zero are kept (gold), then 100 fresh uniform points are added. The gold points follow the predicted outline and the uncertain gift box.</figcaption>
</figure>

<details class="proof">
<summary>Is a point-sampled loss unbiased?</summary>

For a loss that adds up over points, such as BCE, the average over uniform random points is unbiased: averaged over many draws, it equals the full average. Dice is a ratio, so for Dice the evidence is the measured result below. The training points are not an unbiased sample of the dense loss. They deliberately spend the training signal where the prediction is unsure.

</details>

Does sampling hurt accuracy? The paper tried each use on its own:

| matching on | training loss on | AP (COCO) | mIoU (ADE20K) | training memory per image (GB) |
|---|---|---:|---:|---:|
| masks | masks | 41.0 | 45.9 | 18 |
| masks | points | 41.0 | 45.9 | **6** |
| points | masks | 43.1 | **47.3** | 18 |
| points | points (default) | **43.7** | 47.2 | **6** |

Rows 1 and 2 show that training on points instead of whole masks cuts memory per image from 18 GB to 6 GB at equal accuracy. Rows 1 and 3 show that matching on points adds 2.1 AP.

## 7. Smaller changes, and what each part is worth

Three smaller changes to the decoder layer cost no extra computation. Together they improve AP by about 1.4, PQ by 1.1, and mIoU by 0.9.

**Masked attention comes first.** DETR's layer starts with self-attention. Mask2Former's starts with masked attention, so each query has image context before the queries interact.

**Query features are learnable and directly supervised.** DETR starts every query's features at zero and tells the queries apart only by a learned vector per query, its positional embedding, added before every attention step. Mask2Former keeps that embedding but also learns the starting features $\mathbf{X}_0$, and supervises their masks before the decoder runs, which gives the first layer a useful gate. Learning $\mathbf{X}_0$ raises mIoU, and supervising it raises AP and PQ. The paper compares these first predictions to region proposals [[Ren et al. 2015](#ref-ren2015)] that the decoder refines.

**Dropout is removed.** Dropout randomly zeroes activations during training. Taking it out of the decoder improves AP and PQ.

**A new training recipe.** On COCO, Mask2Former trains for 50 epochs (passes over the training data) at batch 16, where MaskFormer used 300 at batch 64, with large-scale jittering (random rescaling between 0.1 and 2 times) and the loss weights of §4.2. The full comparison is folded below.

<details class="proof">
<summary>The two training recipes, side by side</summary>

Both use the AdamW optimizer [[Loshchilov & Hutter 2019](#ref-loshchilov2019)]. In the decoder row, SA, CA, and MA abbreviate self-attention, cross-attention, and masked attention.

| setting | MaskFormer | Mask2Former |
|---|---|---|
| optimizer | AdamW, learning rate $10^{-4}$ | AdamW, learning rate $10^{-4}$ |
| weight decay | $10^{-4}$ | 0.05 |
| schedule (COCO) | 300 epochs at batch 64 | 50 epochs at batch 16, lr ×0.1 at 90% and 95% of steps |
| augmentation | standard scale and crop | LSJ (large-scale jittering) [[Ghiasi et al. 2021](#ref-ghiasi2021)], scale 0.1 to 2.0 with a fixed $1024^2$ crop |
| mask loss | focal ($\lambda{=}20$) + Dice ($\lambda{=}1$), dense | BCE ($\lambda{=}5$) + Dice ($\lambda{=}5$) on 12,544 points |
| $\lambda_{\text{cls}}$ | 1.0 | 2.0 |
| decoder | 6 layers, SA→CA→FFN, dropout 0.1, stride 32 only, zero-init queries | 9 layers, MA→SA→FFN, no dropout, strides {32,16,8}×3, learnable supervised queries |

</details>

**What each part is worth.** The paper takes each part out of the full model, one at a time, and measures the drop. All of these runs use the R50 backbone.

| change (removed or varied) | ΔAP (COCO) | ΔPQ (COCO) | ΔmIoU (ADE20K) |
|---|---:|---:|---:|
| remove masked attention | **−5.9** | **−4.8** | −1.7 |
| replace point matching with dense masks* | −2.7 | −1.1 | −1.3 |
| remove multi-scale high-res features | −2.2 | −1.7 | −1.1 |
| replace supervised learnable features with zero-init | −0.8 | −0.7 | **−1.8** |
| restore dropout | −0.7 | −0.6 | 0.0 |
| standard layer order | −0.5 | −0.3 | −0.9 |

*Measured with point-sampled training loss fixed. The full two-by-two comparison is in §6.2.

The effects overlap, so the rows do not add up. Taken out one at a time, the three layer changes cost 2.0 AP in total, but together they are worth 1.4.

The paper also builds Mask2Former from MaskFormer one step at a time on COCO. Retraining MaskFormer with Mask2Former's recipe raises AP from 34.0 to 37.8. Replacing the decoder raises it to 41.5. Replacing MaskFormer's feature pyramid network (FPN) pixel decoder [[Lin et al. 2017b](#ref-lin2017fpn)] with MSDeformAttn reaches 43.7. The steps interact, so these gains depend on the order and are not fixed shares of the total.

## 8. Results and limitations

### 8.1 Results

The paper's largest models use a Swin-L backbone. The COCO models also use 200 queries and train for 100 epochs, twice the default.

| task / dataset | Mask2Former (Swin-L) | compared with | margin |
|---|---|---|---|
| Panoptic, COCO val | **57.8 PQ** | MaskFormer 52.7, <span style="white-space: nowrap">K-Net 54.6</span> | +5.1 / +3.2 |
| Instance, COCO val | **50.1 AP** | [Swin](#ref-liu2021)-[HTC++](#ref-chen2019) 49.5 | +0.6 |
| Semantic, ADE20K val | **57.7 mIoU** | [BEiT](#ref-bao2022) 57.0 | +0.7 at less than half the parameters |

The ADE20K result uses inference at several image sizes and a FaPN pixel decoder [[Huang et al. 2021](#ref-huang2021)] instead of the default. The COCO results use one size. On COCO's test set, Mask2Former's 50.5 AP beats Swin-HTC++ at one size, 50.2, but not at several sizes, 51.1.

With a ResNet-50 backbone, Mask2Former reaches 51.9 PQ after 50 epochs, where MaskFormer reaches 46.5 after 300. MaskFormer's epochs used batches four times larger, and the paper finds it does about as well in 75 epochs at Mask2Former's batch of 16. Augmentation and losses differ too, so this is not a controlled test of training speed.

The gain is not just scale. Swin-T Mask2Former reaches 53.2 PQ at 232 GFLOPs, above Swin-L MaskFormer's 52.7 at 792 GFLOPs.

### 8.2 Limitations

**Universal means one architecture, not one checkpoint.** A model trained on panoptic labels usually does slightly worse than the same model trained for one task: 41.7 against 43.7 AP on COCO instances, and 46.1 against 47.2 mIoU on ADE20K. Panoptic training does not yet replace task-specific training.

**Small objects remain weaker.** On COCO's validation set, Mask2Former's AP on small objects, under $32^2$ pixels, is 29.9, below Swin-HTC++ at 31.0, although it leads overall.

**It is slower, and the query count depends on the task.** Mask2Former runs at 8.6 frames per second against MaskFormer's 17.6 for the ResNet-50 panoptic models. In the paper's tests, 100 queries work best for instance and semantic segmentation, while 200 help panoptic segmentation slightly.

## 9. Conclusion

Mask2Former works because its changes form a feedback loop. Matching assigns one target to each query. Deep supervision trains every intermediate mask. Each mask restricts the next cross-attention step, and point sampling makes this repeated supervision affordable.

Later models build on the same design: Mask2Former for video [[Cheng et al. 2021b](#ref-cheng2021vis)], **OneFormer** [[Jain et al. 2023](#ref-jain2023)], which trains one model for all three tasks, and **Mask DINO** [[Li et al. 2023](#ref-li2023)], which adds DETR-style detection. Mask2Former's lasting contribution is this query-to-mask refinement loop, more than any one benchmark result.

## Appendix A. Implementation notes

The reference implementation is [`facebookresearch/Mask2Former`](https://github.com/facebookresearch/Mask2Former), built on Detectron2 [[Wu et al. 2019](#ref-wu2019)]. Hugging Face provides it as `Mask2FormerForUniversalSegmentation`.

<details class="proof">
<summary>The decoder layer in code, and the code's names for the losses</summary>

The reference repository was archived on January 1, 2025. In Hugging Face, `Mask2FormerImageProcessor` or `AutoImageProcessor` exposes semantic, instance, and panoptic post-processing methods. The decoder core is shown below in simplified PyTorch-style pseudocode.

```python
def decoder_layer(x, feats_l, size_l, pos_l, lvl_emb, query_pos, prev_mask_logits, pixel_embeddings, num_heads):
    # x: (N,B,C) · feats_l: (H_l*W_l,B,C) · size_l: (H_l,W_l)
    m = F.interpolate(prev_mask_logits, size=size_l, mode="bilinear", align_corners=False)
    attn_mask = (m.sigmoid() < 0.5).flatten(2)  # (B,N,H_l*W_l)
    attn_mask = attn_mask[:, None].repeat(1, num_heads, 1, 1).flatten(0, 1).detach()  # (B*num_heads,N,H_l*W_l)
    attn_mask[attn_mask.all(dim=-1)] = False  # Prevent all-masked softmax.

    # The scale embedding is part of both the key and value memory.
    feats_l = feats_l + lvl_emb
    x = norm1(x + cross_attn(q=with_pos(x, query_pos), k=with_pos(feats_l, pos_l), v=feats_l, attn_mask=attn_mask))
    x = norm2(x + self_attn(q=with_pos(x, query_pos), k=with_pos(x, query_pos), v=x))
    x = norm3(x + ffn(x))

    h = decoder_norm(x).transpose(0, 1)
    cls_logits = class_head(h)
    mask_logits = einsum("bnc,bchw->bnhw", mask_head(h), pixel_embeddings)
    return x, cls_logits, mask_logits
```

The heads run once on $\mathbf{X}_0$ before the loop and after each of nine decoder layers. The initial prediction supplies $Z_0$. The final output receives the main loss, and the preceding nine outputs receive auxiliary losses.

Matching calls [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html), a modified Jonker-Volgenant algorithm, on the rectangular predictions-by-targets matrix of real targets, which is equivalent to the padded square problem of §4.1. In a minibatch of $B$ images, the class loss takes its weighted mean over all $BN$ query slots.

The code's names differ from this article's. The released criterion logs the class loss as `loss_ce` and the mask BCE, this article's $\mathcal{L}_{\text{ce}}$, as `loss_mask`.

</details>

## References

1. <a name="ref-cheng2022"></a>Cheng, Misra, Schwing, Kirillov, Girdhar. "Masked-attention Mask Transformer for Universal Image Segmentation." 2022. [arXiv:2112.01527](https://arxiv.org/abs/2112.01527) · [Code](https://github.com/facebookresearch/Mask2Former)
2. <a name="ref-cheng2021"></a>Cheng, Schwing, Kirillov. "Per-Pixel Classification is Not All You Need for Semantic Segmentation." 2021. [arXiv:2107.06278](https://arxiv.org/abs/2107.06278)
3. <a name="ref-carion2020"></a>Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko. "End-to-End Object Detection with Transformers." 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
4. <a name="ref-kirillov2019pan"></a>Kirillov, He, Girshick, Rother, Dollár. "Panoptic Segmentation." 2019. [arXiv:1801.00868](https://arxiv.org/abs/1801.00868)
5. <a name="ref-kirillov2020"></a>Kirillov, Wu, He, Girshick. "PointRend: Image Segmentation as Rendering." 2020. [arXiv:1912.08193](https://arxiv.org/abs/1912.08193)
6. <a name="ref-zhu2021"></a>Zhu, Su, Lu, Li, Wang, Dai. "Deformable DETR: Deformable Transformers for End-to-End Object Detection." 2021. [arXiv:2010.04159](https://arxiv.org/abs/2010.04159)
7. <a name="ref-zhang2021"></a>Zhang, Pang, Chen, Loy. "K-Net: Towards Unified Image Segmentation." 2021. [arXiv:2106.14855](https://arxiv.org/abs/2106.14855)
8. <a name="ref-wang2021"></a>Wang, Zhu, Adam, Yuille, Chen. "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers." 2021. [arXiv:2012.00759](https://arxiv.org/abs/2012.00759)
9. <a name="ref-he2017"></a>He, Gkioxari, Dollár, Girshick. "Mask R-CNN." 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
10. <a name="ref-long2015"></a>Long, Shelhamer, Darrell. "Fully Convolutional Networks for Semantic Segmentation." 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
11. <a name="ref-vaswani2017"></a>Vaswani et al. "Attention Is All You Need." 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
12. <a name="ref-kuhn1955"></a>Kuhn. "The Hungarian Method for the Assignment Problem." 1955. [doi:10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109)
13. <a name="ref-munkres1957"></a>Munkres. "Algorithms for the Assignment and Transportation Problems." 1957. [doi:10.1137/0105003](https://doi.org/10.1137/0105003)
14. <a name="ref-milletari2016"></a>Milletari, Navab, Ahmadi. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." 2016. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
15. <a name="ref-lin2017"></a>Lin, Goyal, Girshick, He, Dollár. "Focal Loss for Dense Object Detection." 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
16. <a name="ref-lin2017fpn"></a>Lin, Dollár, Girshick, He, Hariharan, Belongie. "Feature Pyramid Networks for Object Detection." 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
17. <a name="ref-gao2021"></a>Gao, Zheng, Wang, Dai, Li. "Fast Convergence of DETR with Spatially Modulated Co-Attention." 2021. [arXiv:2101.07448](https://arxiv.org/abs/2101.07448)
18. <a name="ref-sun2021"></a>Sun, Cao, Yang, Kitani. "Rethinking Transformer-based Set Prediction for Object Detection." 2021. [arXiv:2011.10881](https://arxiv.org/abs/2011.10881)
19. <a name="ref-liu2021"></a>Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
20. <a name="ref-he2016"></a>He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition." 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
21. <a name="ref-chen2019"></a>Chen et al. "Hybrid Task Cascade for Instance Segmentation." 2019. [arXiv:1901.07518](https://arxiv.org/abs/1901.07518)
22. <a name="ref-ren2015"></a>Ren, He, Girshick, Sun. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
23. <a name="ref-huang2021"></a>Huang, Lu, Cheng, He. "FaPN: Feature-Aligned Pyramid Network for Dense Image Prediction." 2021. [arXiv:2108.07058](https://arxiv.org/abs/2108.07058)
24. <a name="ref-loshchilov2019"></a>Loshchilov, Hutter. "Decoupled Weight Decay Regularization." 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
25. <a name="ref-ghiasi2021"></a>Ghiasi et al. "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation." 2021. [arXiv:2012.07177](https://arxiv.org/abs/2012.07177)
26. <a name="ref-bao2022"></a>Bao, Dong, Wei. "BEiT: BERT Pre-Training of Image Transformers." 2022. [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
27. <a name="ref-everingham2015"></a>Everingham et al. "The PASCAL Visual Object Classes Challenge: A Retrospective." 2015. [doi:10.1007/s11263-014-0733-5](https://doi.org/10.1007/s11263-014-0733-5)
28. <a name="ref-lin2014"></a>Lin et al. "Microsoft COCO: Common Objects in Context." 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
29. <a name="ref-zhou2017"></a>Zhou et al. "Scene Parsing through ADE20K Dataset." 2017. [Project page](https://ade20k.csail.mit.edu/)
30. <a name="ref-cheng2021vis"></a>Cheng et al. "Mask2Former for Video Instance Segmentation." 2021. [arXiv:2112.10764](https://arxiv.org/abs/2112.10764)
31. <a name="ref-jain2023"></a>Jain et al. "OneFormer: One Transformer to Rule Universal Image Segmentation." 2023. [arXiv:2211.06220](https://arxiv.org/abs/2211.06220)
32. <a name="ref-li2023"></a>Li et al. "Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation." 2023. [arXiv:2206.02777](https://arxiv.org/abs/2206.02777)
33. <a name="ref-wu2019"></a>Wu, Kirillov, Massa, Lo, Girshick. "Detectron2." 2019. [Code](https://github.com/facebookresearch/detectron2)

## Citation

Massih, Peter. "Mask2Former, Dissected." *peteramassih.com*, Jul 2026. https://peteramassih.com/writing/mask2former/.

```bibtex
@article{massih2026mask2former,
  title   = {Mask2Former, Dissected},
  author  = {Massih, Peter},
  journal = {peteramassih.com},
  year    = {2026},
  month   = {July},
  url     = {https://peteramassih.com/writing/mask2former/}
}
```
