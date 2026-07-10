---
title: "Mask2Former, Dissected: One Transformer to Segment Them All"
description: "A ground-up walk through Mask2Former (CVPR 2022): the lineage from FCNs to set prediction, every equation derived, the training recipe, and the ablations ranked by what they bought."
pubDate: 2026-07-07
tags: [computer-vision, segmentation, transformers, paper-dissection]
math: true
series: "Peter's Patches"
part: 1
---

**TL;DR.** Mask2Former [[Cheng et al. 2022](#ref-cheng2022)] is one architecture that handles all three segmentation tasks, panoptic, instance, and semantic, and beats the models built specially for each. Its largest model, with a Swin-L backbone, reaches 57.8 PQ and 50.1 AP on COCO and 57.7 mIoU on ADE20K, and at the smaller ResNet-50 scale it trains six times faster than its predecessor. The trick is not scale. It is a rewired Transformer decoder whose cross-attention is masked to each query's own predicted foreground, a coarse-to-fine feeding schedule, three cheap optimization changes, and a point-sampled loss that cuts training memory threefold. This post works through the paper from the ground up, tracing where each idea came from, deriving the equations rather than stating them, and ranking the ablations by what they actually bought.

**Contents.** [0. Background](#0-the-background-you-need) · [1. The problem, formally](#1-the-segmentation-problem-formally) · [2. Two paradigms](#2-origins-two-paradigms) · [3. Set prediction](#3-the-set-prediction-machinery) · [4. The meta-architecture](#4-the-meta-architecture) · [5. Masked attention](#5-masked-attention) · [6. Multi-scale features](#6-feeding-the-decoder) · [7. Decoder rewiring](#7-rewiring-the-decoder-layer) · [8. Point-sampled losses](#8-losses-match-on-points-train-on-points) · [9. The recipe](#9-the-full-training-recipe) · [10. Results](#10-results-worth-remembering) · [11. Ablations, ranked](#11-what-actually-mattered) · [12. Limitations](#12-limitations-read-honestly) · [13. Lineage forward](#13-where-it-went-next) · [14. Implementation](#14-implementation-corner) · [15. Test yourself](#15-test-yourself) · [16. References](#16-references) · [17. Citation](#17-citation)

## 0. The background you need

Start here if any of this is new. Skip to §1 if words like attention, embedding, and loss already feel comfortable. The notation table below is a reference to return to, not something to memorize.

**Images, pixels, and the job.** An image is a grid of pixels. Segmentation decides which pixels belong together and what the resulting groups are. §1 makes the three meanings of "belong together" precise.

**Feature maps.** A backbone network converts the image into a much smaller grid where each cell holds a vector summarizing a whole patch of the original. That grid is a feature map. Stride 32, written $1/32$, means the grid is 32 times smaller per side, so a 1024 by 1024 image becomes a 32 by 32 summary. One consequence matters most. A small, distant object can vanish at that scale, because no cell is left to represent it. That single fact drives all of §6.

**Convolutions.** The backbone is built mostly from convolutions, the core operation of vision networks. A filter is a small grid of learned weights. Slide it across the image, and at each position take the dot product of the filter with the patch of pixels beneath it (multiply matching cells, add the results). That one number becomes a pixel of the output, so a single filter sweeps the whole image and produces a new map that lights up wherever its pattern appears. Layers of such filters, stacked with nonlinearities between them, make a convolutional network (CNN), and a $1\times1$ filter is the special case that mixes only the values sitting at one pixel, without looking at its neighbors.

<figure class="viz">
<svg class="m2f-conv" style="min-width: 560px" viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="One convolution step: a 3 by 3 filter of weights lines up with a 3 by 3 patch of the input, multiplies cell by cell, and sums the nine products into a single pixel of the output feature map. Here a vertical-edge filter meets a vertical edge and returns a strong 3.0.">
<defs>
<style>
.cv-lbl { font-family: Geist, ui-sans-serif, system-ui, sans-serif; fill: #171717; font-size: 13px; }
.cv-sub { font-family: Geist, ui-sans-serif, system-ui, sans-serif; fill: #6b6b6b; font-size: 11px; }
.cv-num { font-family: Geist, ui-sans-serif, system-ui, sans-serif; fill: #52525b; font-size: 11px; }
</style>
<marker id="cv-ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#9a9a9a"/></marker>
</defs>
<rect width="700" height="240" rx="6" fill="#fafafa"/>
<text class="cv-lbl" x="93" y="38" text-anchor="middle">input</text>
<rect x="28" y="52" width="130" height="130" fill="none" stroke="#9a9a9a"/>
<line x1="54" y1="52" x2="54" y2="182" stroke="#d4d4d4"/><line x1="80" y1="52" x2="80" y2="182" stroke="#d4d4d4"/><line x1="106" y1="52" x2="106" y2="182" stroke="#d4d4d4"/><line x1="132" y1="52" x2="132" y2="182" stroke="#d4d4d4"/>
<line x1="28" y1="78" x2="158" y2="78" stroke="#d4d4d4"/><line x1="28" y1="104" x2="158" y2="104" stroke="#d4d4d4"/><line x1="28" y1="130" x2="158" y2="130" stroke="#d4d4d4"/><line x1="28" y1="156" x2="158" y2="156" stroke="#d4d4d4"/>
<rect x="28" y="52" width="78" height="78" fill="#b8860b" fill-opacity="0.10" stroke="#b8860b" stroke-width="2"/>
<g class="cv-num">
<text x="41" y="70" text-anchor="middle">0</text><text x="67" y="70" text-anchor="middle">.5</text><text x="93" y="70" text-anchor="middle">1</text>
<text x="41" y="96" text-anchor="middle">0</text><text x="67" y="96" text-anchor="middle">.5</text><text x="93" y="96" text-anchor="middle">1</text>
<text x="41" y="122" text-anchor="middle">0</text><text x="67" y="122" text-anchor="middle">.5</text><text x="93" y="122" text-anchor="middle">1</text>
</g>
<rect x="54" y="52" width="78" height="78" fill="none" stroke="#9a9a9a" stroke-dasharray="3 3" opacity="0.55"/>
<text class="cv-sub" x="145" y="200" text-anchor="middle">slide &#8594;</text>
<text class="cv-lbl" x="275" y="38" text-anchor="middle">3&#215;3 filter</text>
<rect x="236" y="52" width="78" height="78" fill="#b8860b" fill-opacity="0.13" stroke="#b8860b"/>
<line x1="262" y1="52" x2="262" y2="130" stroke="#d4d4d4"/><line x1="288" y1="52" x2="288" y2="130" stroke="#d4d4d4"/>
<line x1="236" y1="78" x2="314" y2="78" stroke="#d4d4d4"/><line x1="236" y1="104" x2="314" y2="104" stroke="#d4d4d4"/>
<g class="cv-num">
<text x="249" y="70" text-anchor="middle">&#8722;1</text><text x="275" y="70" text-anchor="middle">0</text><text x="301" y="70" text-anchor="middle">+1</text>
<text x="249" y="96" text-anchor="middle">&#8722;1</text><text x="275" y="96" text-anchor="middle">0</text><text x="301" y="96" text-anchor="middle">+1</text>
<text x="249" y="122" text-anchor="middle">&#8722;1</text><text x="275" y="122" text-anchor="middle">0</text><text x="301" y="122" text-anchor="middle">+1</text>
</g>
<line x1="106" y1="91" x2="234" y2="91" stroke="#9a9a9a" stroke-dasharray="4 3"/>
<text class="cv-sub" x="388" y="66" text-anchor="middle">&#8857; then &#931;</text>
<text class="cv-num" x="388" y="120" text-anchor="middle">sum of 9</text>
<line x1="316" y1="91" x2="460" y2="66" stroke="#9a9a9a" marker-end="url(#cv-ar)"/>
<text class="cv-lbl" x="501" y="38" text-anchor="middle">feature map</text>
<rect x="462" y="52" width="78" height="78" fill="none" stroke="#9a9a9a"/>
<line x1="488" y1="52" x2="488" y2="130" stroke="#d4d4d4"/><line x1="514" y1="52" x2="514" y2="130" stroke="#d4d4d4"/>
<line x1="462" y1="78" x2="540" y2="78" stroke="#d4d4d4"/><line x1="462" y1="104" x2="540" y2="104" stroke="#d4d4d4"/>
<rect x="462" y="52" width="26" height="26" fill="#0d9488"/>
<text x="475" y="69" text-anchor="middle" fill="#ffffff" font-family="Geist, ui-sans-serif, system-ui, sans-serif" font-size="12px">3.0</text>
<text class="cv-sub" x="501" y="152" text-anchor="middle">one patch &#8594; one pixel</text>
<text class="cv-sub" x="350" y="224" text-anchor="middle">the same nine weights sweep every position, lighting up wherever the filter's pattern appears</text>
</svg>
<figcaption>Fig. 0. One convolution step. The gold filter lines up with a gold patch of the input, multiplies cell by cell, and sums the nine products into a single output pixel. Slide the filter one cell over, to the dashed box, and repeat to fill the whole map. This filter is a vertical-edge detector, so meeting a vertical edge in the input returns a strong response.</figcaption>
</figure>

**Embeddings and the dot product.** The summary vectors are embeddings, vectors whose geometry encodes meaning, arranged so similar things point in similar directions. The dot product measures how similar two of them are, and the intuition is one line of algebra. For vectors $u, v$ with angle $\theta$ between them,

$$
u^\top v = \sum_k u_k v_k = \|u\|\,\|v\|\cos\theta,
$$

so it is large and positive when the vectors point the same way, exactly zero when they are orthogonal, and negative when they disagree. Keep this close, because the architecture's central trick (§4) is simple. A segment's mask is nothing but the dot product between one vector and every pixel's vector, squashed into $[0,1]$.

**Softmax.** Given scores $z_1, \dots, z_n$ (pre-softmax scores are called logits, a word that recurs below), softmax turns them into weights

$$
w_i = \frac{e^{z_i}}{\sum_j e^{z_j}},
$$

which are positive and sum to 1, a probability distribution over options. Two properties matter here, and both are visible in the formula. First, $e^{z_i} > 0$ always, so softmax never outputs an exact zero. Every option keeps a sliver of weight, however unpromising. Second, whatever the scores, the weights always sum to 1. The first property causes the central pathology of this paper (§5.1). The second is what the fix exploits (§5.2). Sending a forbidden entry's score to $-\infty$ drops it out, and the survivors renormalize among themselves.

**Attention.** Attention is a soft, differentiable dictionary lookup. A query asks a question. Every location in the image offers a key, a description the question is matched against, and a value, which is what the location contains. Score each location by the dot product of query and key, softmax the scores into weights $w_x$, and return the weighted average $\sum_x w_x\, v_x$ of the values. In matrices, stack the $N$ query vectors as rows of $Q \in \mathbb{R}^{N\times d}$ and the $n$ locations' keys and values as rows of $K, V \in \mathbb{R}^{n\times d}$:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V.
$$

Read it factor by factor. $QK^\top$ is an $N \times n$ table of every question dotted with every key. The softmax runs along each row, turning it into a probability distribution over locations. Multiplying by $V$ then takes each row's weighted average of the value vectors. The $\sqrt d$ is a temperature. A dot product sums $d$ terms, so its typical magnitude grows like $\sqrt d$, and without the division long vectors would push the softmax into saturation, where it behaves like a hard argmax (all the weight on the single highest score) and gradients die. When queries read from the image it is called cross-attention, with $Q$ built from the queries and $K, V$ from the image. When one set of tokens reads from itself it is called self-attention, with $Q = XW_Q$, $K = XW_K$, $V = XW_V$ all linear views of the same tokens $X$ through three learned matrices, so every token's update is a weighted average of the others, the weights set by their mutual dot products.

**Transformers and decoders.** A Transformer layer is attention plus a small per-token network, wrapped in residual connections. As a function it is just two updates applied in order:

$$
x \,\leftarrow\, x + \operatorname{Attention}(x,\ \text{context}), \qquad
x \,\leftarrow\, x + \operatorname{FFN}(x).
$$

The attention step mixes in information from wherever this layer is allowed to look (the image for cross-attention, the other queries for self-attention). The feed-forward network $\operatorname{FFN}$, a two-layer MLP applied to each token separately, processes what was just read. An MLP (multi-layer perceptron) is a plain function $\operatorname{MLP}(x) = W_2\,\phi(W_1 x + b_1) + b_2$, two affine maps, each a matrix multiply plus a bias, with a pointwise nonlinearity $\phi$ between them. Several nonlinearities are used in practice, and we write $\phi$ as the common ReLU, $\phi(t) = \max(0, t)$, only for concreteness. Drop $\phi$ entirely and the two maps collapse into one affine map $x \mapsto W_2 W_1 x + (W_2 b_1 + b_2)$, no more expressive than a single layer, so the nonlinearity is the whole source of an MLP's power. And because both results are added onto $x$ rather than replacing it, a layer can refine what a token knows without having to overwrite it. The decoder in this post is a stack of such layers in which 100 learned query tokens repeatedly read from the image and confer among themselves. After nine layers, each query has become a description of one segment.

**Losses, gradients, training.** A loss $\mathcal{L}$ is a single number measuring how wrong the current output is. Training repeats one step. Compute the gradient $\nabla_w \mathcal{L}$, the direction in weight space that increases the loss fastest, then move every weight a small step the other way, $w \leftarrow w - \eta\, \nabla_w \mathcal{L}$, where $\eta$ is the step size, the learning rate. A network is shaped less by its wiring than by what you choose to measure, and most of this paper's cleverness lives in the measuring (§3, §8).

**IoU.** Intersection over union for two regions $A$ and $B$:

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|},
$$

which is 1 when they are identical and 0 when they are disjoint. This is the standard way to ask whether two regions are the same, and it sits inside every metric in §1.

**Sets versus lists.** A list has an order. A set does not. The model's 100 guesses come out in arbitrary order, and a fair grader must not care about that order. This innocent-sounding fact forces all of the machinery in §3.

**How to read this post.** Every heavy section opens in plain words before any symbol appears. Proofs are short, boxed off, and skippable on a first pass. Read it twice, once for the story and once for the math.

## Notation

| symbol | meaning |
|---|---|
| $I \in \mathbb{R}^{3\times H\times W}$ | input image. $H_l \times W_l$ is the size of the feature map used at decoder layer $l$ |
| $\Omega$ | the pixel grid, the set of pixel locations of the map under discussion. $\lvert\Omega\rvert$ is its size |
| $C$ | shared feature dimension of queries and image features (256 in the reference implementation) |
| $N$ | number of object queries (100 by default, 200 for the largest models) |
| $K$, $\varnothing$ | number of classes, and the "no-object" class appended to them |
| $(m_i, c_i)$ | prediction $i$: a soft mask $m_i \in [0,1]^{H\times W}$ and a class $c_i \in \{1,\dots,K\}\cup\{\varnothing\}$ |
| $\hat p_i(c)$ | predicted probability that query $i$ has class $c$ |
| $\mathbf{X}_l \in \mathbb{R}^{N\times C}$ | query features after decoder layer $l$. $\mathbf{X}_0$ are the learnable input query features |
| $\mathbf{Q}_l, \mathbf{K}_l, \mathbf{V}_l$ | attention projections. $\mathbf{K}_l,\mathbf{V}_l \in \mathbb{R}^{H_lW_l\times C}$ come from image features |
| $M_{l}$ | mask predictions of layer $l$, resized as logits to the next layer's attention resolution, then binarized at $0.5$ |
| $\mathcal{M}_{l}$ | the additive attention mask built from $M_{l}$. It is $0$ on foreground and $-\infty$ elsewhere |
| $\mathcal{E}_{\text{pixel}}$ | per-pixel embeddings from the pixel decoder, at stride 4 |
| $\sigma(\cdot)$ | the logistic sigmoid $\sigma(z) = 1/(1+e^{-z})$, which squashes a real score into $[0,1]$ |
| $\lambda_{\text{ce}}, \lambda_{\text{dice}}, \lambda_{\text{cls}}$ | weights on the BCE, Dice, and classification losses: $\lambda_{\text{ce}} = \lambda_{\text{dice}} = 5.0$ and $\lambda_{\text{cls}} = 2.0$ ($0.1$ for the no-object class $\varnothing$) |
| $K_{\text{pt}}$ | number of sampled points for mask losses: $12{,}544 = 112^2$ |

## 1. The segmentation problem, formally

Image segmentation asks which pixels belong together. The interesting part is that "belong together" admits several meanings. Every dataset fixes a set of categories $\mathcal{C}$, the labels it knows: car, person, road, sky. Vision researchers split those labels into two kinds, following [[Kirillov et al. 2019](#ref-kirillov2019pan)]. *Things* are countable objects with identities, like cars and people, where "which car is this pixel part of" is a meaningful question. *Stuff* is amorphous material, like road and sky, where counting makes no sense. There is no sky number two. Writing $\mathcal{C}_{\text{th}}$ for the thing categories and $\mathcal{C}_{\text{st}}$ for the stuff categories, the full set is $\mathcal{C} = \mathcal{C}_{\text{th}} \sqcup \mathcal{C}_{\text{st}}$, where $\sqcup$ just means every category is one or the other, never both. The three tasks are then three output spaces over the same pixels, differing exactly in how they treat the two kinds.

**Semantic segmentation** is a map $f: \Omega \to \mathcal{C}$ on the pixel grid $\Omega$. One label per pixel, no identities, so two adjacent cars fuse into one car region. Its metric averages region overlap per class:

$$
\text{mIoU} = \frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}} \frac{|P_c \cap G_c|}{|P_c \cup G_c|},
$$

where $P_c$ and $G_c$ are the predicted and ground-truth pixel sets of class $c$ [[Everingham et al. 2015](#ref-everingham2015)].

**Instance segmentation** outputs a set of scored masks over things only, $\{(m_i, c_i, s_i)\}$ with confidence score $s_i$, evaluated by mask AP. For each class, predictions are ranked by score. A prediction counts as a true positive when its mask IoU with an unclaimed ground truth meets or exceeds a threshold $\tau$, and a false positive otherwise. Precision is the fraction of kept predictions that are true positives, recall the fraction of ground truths recovered, and the precision-recall curve plots one against the other as the score threshold sweeps from high to low. AP is the area under that curve, averaged over ten IoU thresholds in the COCO style [[Lin et al. 2014](#ref-lin2014)]:

$$
P = \frac{|\mathit{TP}|}{|\mathit{TP}| + |\mathit{FP}|}, \qquad
R = \frac{|\mathit{TP}|}{|G|}, \qquad
\text{AP} = \frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}} \int_0^1 p_\tau(r)\,\mathrm{d}r,
$$

where $|G|$ is the number of ground-truth masks in the class, $\mathcal{T} = \{0.50, 0.55, \dots, 0.95\}$ is the threshold sweep, and $p_\tau(r)$ is the precision at recall $r$ under threshold $\tau$. COCO reports the mean of this per-class AP over all categories. The metric quietly demands calibrated ranking, confidence scores that actually track how good each mask is, not just good masks. That will matter in post-processing (§9).

**Panoptic segmentation** [[Kirillov et al. 2019](#ref-kirillov2019pan)] unifies both. Every pixel receives a class and an instance id, with identities on things and plain categories on stuff. Predicted and ground-truth segments are matched, writing $\mathit{TP}$ for the matched pairs (true positives), $\mathit{FP}$ for predictions that match nothing (false positives), and $\mathit{FN}$ for ground-truth segments left unmatched (false negatives). Quality is

$$
\text{PQ} = \frac{\sum_{(p,g)\in \mathit{TP}}\text{IoU}(p,g)}{|\mathit{TP}| + \tfrac12|\mathit{FP}| + \tfrac12|\mathit{FN}|}
=
\underbrace{\frac{\sum_{(p,g)\in \mathit{TP}}\text{IoU}(p,g)}{|\mathit{TP}|}}_{\text{SQ}}
\times
\underbrace{\frac{|\mathit{TP}|}{|\mathit{TP}| + \tfrac12|\mathit{FP}| + \tfrac12|\mathit{FN}|}}_{\text{RQ}},
$$

where a match requires $\text{IoU} > 0.5$. PQ is computed per class this way, then averaged over the categories in $\mathcal{C}$, the same class averaging mIoU uses. The factorization into segmentation quality (SQ), the average IoU over the matched pairs, and recognition quality (RQ), the share of segments correctly detected at all, is immediate (multiply and divide by $|\mathit{TP}|$). The matching rule hides a small theorem that makes PQ well defined in the first place.

**Lemma (matches are unique).** *If predicted segments and ground-truth segments are each pairwise disjoint, as the panoptic format requires, then $\text{IoU} > 0.5$ pairs one-to-one: each ground-truth segment matches at most one prediction, and each prediction at most one ground truth.*

**Proof.** Suppose a prediction $p$ has $\text{IoU}(p, g) > \tfrac12$. Then

$$
\begin{aligned}
|p \cap g|
&> \tfrac12\,|p \cup g| && \text{definition of IoU, rearranged}\\[2pt]
&\ge \tfrac12\,|g| && \text{since } g \subseteq p \cup g.
\end{aligned}
$$

So every matching prediction covers more than half of $g$. Now suppose two predictions $p_1, p_2$ both matched $g$. The panoptic format keeps predictions disjoint, so $p_1 \cap g$ and $p_2 \cap g$ are disjoint subsets of $g$, each larger than $\tfrac12|g|$. Together they would hold more than $|g|$ pixels inside $g$, which is impossible. So at most one prediction matches $g$. The same argument runs the other way. Since the panoptic format also keeps the ground-truth segments disjoint, and $p \subseteq p \cup g$ gives $|p \cap g| > \tfrac12|p|$, each prediction matches at most one ground truth. $\blacksquare$

Above the $0.5$ threshold, matching is therefore unambiguous, and greedy matching, pairing segments one at a time with the best still-unclaimed partner, is exact. Contrast this with training-time matching (§3), where predictions overlap freely, costs are soft, and a genuine assignment problem appears.

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: one scene regrouped under the three segmentation semantics">
<source data-src="/assets/m2f/query_becomes_segment.webm" type="video/webm">
<source data-src="/assets/m2f/query_becomes_segment.mp4" type="video/mp4">
</video>
<figcaption>Fig. 1. Gold circles are the model's query slots, and the blue-gray panel is the image, with two ducks and a dog. Each query starts with a rough guess at where its object is, the soft gold region, and the nine tick marks are the decoder layers that sharpen those guesses until each claims one object. Two queries briefly contend for the same duck until one backs off. That is self-attention keeping them from colliding. The triangle is the class head. It weighs the candidate classes, including a hollow one for no-object, keeps the most likely, and colors the query with it, teal for duck, deep gold for dog. One leftover query ends on the hollow no-object choice, so its slot empties, and the unused queries fade with it. So the region answers where, the color answers what. At the end the same predictions regroup three ways: semantic labels every pixel by class, the two ducks sharing one hue, instance keeps them as separate identities with outlines but drops the background, and panoptic keeps both, the per-duck identities and the labeled background. The model runs once, and only the grouping changes.</figcaption>
</figure>

These tasks differ only in the semantics of grouping. That observation motivates the whole research program. Yet by 2021 each had its own architecture family, its own tricks, and its own hardware optimizations. Three times the effort, and specializations that do not transfer, as §6.3's pixel-decoder comparison will show.

## 2. Origins: two paradigms

### 2.1 Per-pixel classification (2015 onward)

The fully convolutional network (FCN) of [[Long et al. 2015](#ref-long2015)] recast semantic segmentation as dense classification. A convolutional network ends in a $1{\times}1$ classifier head, one filter per class scoring each pixel's feature vector on its own, and outputs $\hat y \in \mathbb{R}^{|\mathcal{C}|\times H\times W}$, trained with per-pixel cross-entropy

$$
\mathcal{L} = -\frac{1}{|\Omega|}\sum_{x\in\Omega}\log \hat p_x(g_x),
$$

where $\hat p_x$ is the softmax of the class scores at pixel $x$ and $g_x$ is that pixel's true class. The lineage after it is a search for context: dilated convolutions and pyramid pooling in DeepLab and PSPNet [[Chen et al. 2018](#ref-chen2018), [Zhao et al. 2017](#ref-zhao2017)], self-attention variants [[Wang et al. 2018](#ref-wang2018), [Fu et al. 2019](#ref-fu2019)], and finally pure-Transformer per-pixel models like Segmenter and SegFormer [[Strudel et al. 2021](#ref-strudel2021), [Xie et al. 2021](#ref-xie2021)].

The formal reason this family cannot do instances is also the reason universal architectures exist. A per-pixel classifier's output space is $\mathcal{C}^{\Omega}$, a fixed product of per-pixel labels. Instance segmentation outputs a set of variable size whose elements carry identities that are pure bookkeeping. Swapping the names "car 1" and "car 2" gives the same answer. A function into $\mathcal{C}^\Omega$ has no variable for identity at all. Bolting on instance ids as extra channels fails, because any fixed channel-to-identity assignment is arbitrary, and the network would be punished for producing a correct answer in a different order. The output is invariant under any relabeling of the object instances (formally, a symmetric group action, made precise in §3.1), so the loss must be too, and per-pixel losses are not. Detection had solved this around 2015 with anchors, a fixed grid of preset candidate boxes the detector scores and nudges, plus non-maximum suppression, which ranks overlapping detections by score and deletes any that overlap a higher one. Together they impose an artificial order and then de-duplicate, exactly the hand-tuned machinery DETR, the detection Transformer of §2.2, was built to delete.

### 2.2 Mask classification (2017 onward)

The alternative lineage outputs segments directly. **Mask R-CNN** [[He et al. 2017](#ref-he2017)] predicts a binary mask inside each detected bounding box, the axis-aligned rectangle that encloses an object. That is mask classification, but tethered to boxes, which caps it at things and makes stuff awkward. The general query-based form arrived with **DETR** (DEtection TRansformer) [[Carion et al. 2020](#ref-carion2020)]. Represent each potential object as a learned query vector, decode all $N$ of them in parallel, and make the loss order-free via bipartite matching (§3), a one-to-one pairing between the predictions and the targets. DETR was a detector, but its panoptic extension already predicted masks from queries. **MaX-DeepLab** [[Wang et al. 2021](#ref-wang2021)] then made mask prediction fully end-to-end, one network trained by one loss with no hand-built stages in between, for panoptic segmentation specifically. **MaskFormer** [[Cheng et al. 2021](#ref-cheng2021)], by the same first author as Mask2Former, showed the sharp result that a DETR-style mask classifier is not just a way to unify tasks. It beats per-pixel classifiers at semantic segmentation itself. **K-Net** [[Zhang et al. 2021](#ref-zhang2021)] pushed the set-prediction view into instance segmentation with dynamic kernels, convolution filters generated per segment at run time instead of staying fixed after training.

**The idea, before the formalism.** Stop asking every pixel "what class are you" and start asking "which region are you part of." Mask classification splits segmentation into two independent questions per segment: where it is, a binary mask over the pixels, and what it is, one label for the whole region. In probabilistic terms, each slot $i$ carries a field of per-pixel Bernoulli variables, where $m_i(x)$ is the probability that pixel $x$ belongs to segment $i$, together with one categorical distribution $\hat p_i$ over the $K+1$ classes for what the region is. The mask carries no class information and the class carries no location information. That clean separation of where from what is exactly what §4 hard-wires into the architecture, one head per question.

Formally, mask classification predicts

$$
\{(m_i, c_i)\}_{i=1}^{N}, \qquad m_i \in [0,1]^{H\times W},\quad c_i \in \{1,\dots,K\}\cup\{\varnothing\},
$$

and the three tasks fall out by interpretation: segments as categories (semantic), segments as things with identity (instance), or both (panoptic). One architecture, one loss, three annotation schemes.

**Concretely.** Take the puppy in the figure below. The model's whole answer for it is one pair: the class $c_i = \texttt{dog}$, and the mask $m_i$, a grid the size of the image holding one number per pixel, near $1$ inside the dog and near $0$ everywhere else. Coarsen that grid, as the right panel does, and it becomes a matrix of one number per cell. Shrink that down to a handful of cells and it reads:

$$
c_i = \texttt{dog}, \qquad
m_i \;\longrightarrow\;
\begin{pmatrix}
0 & 0 & 1 & 1 & 0\\
0 & 1 & 1 & 1 & 1\\
0 & 1 & 1 & 1 & 0\\
0 & 0 & 1 & 0 & 0
\end{pmatrix}.
$$

The matrix says where, the single label says what, and the two never mix. The model does not actually emit hard ones and zeros. Each entry is a probability $m_i(x) \in [0,1]$ that gets thresholded at $0.5$, which is why the definition above writes $[0,1]^{H\times W}$ and not $\{0,1\}^{H\times W}$.

<figure class="viz">
<img src="/assets/m2f/mask_classification.webp" width="1572" height="568" loading="lazy" decoding="async" alt="Left: a photograph of a French bulldog puppy on a blanket, the puppy filled with a translucent gold overlay tracing the model's predicted mask. Right: the same mask drawn as a coarse grid, gold cells where a pixel belongs to the dog and blank cells elsewhere.">
<figcaption>Fig. 2. A single prediction from the trained model. Left: a photograph, where the gold region is the model's own predicted mask for one query, so gold is what the model asserts. Right: that same mask coarsened to a grid, gold where a pixel belongs to the dog and blank where it does not, which is all a binary mask ever is. The model read the whole image in one pass and returned two pairs, the dog as a thing and the blanket behind it as stuff, exactly the split of <a class="section-ref" href="#1-the-segmentation-problem-formally">§1</a>. Photograph public domain (CC0). Segmentation by Mask2Former with a Swin-L backbone on COCO panoptic.</figcaption>
</figure>

So by late 2021 universal architectures existed, and an awkward fact sat in the Mask2Former introduction. Researchers kept building specialists anyway. Three numbers explain why. Accuracy: the best universal instance result trailed the best specialist by over 9 AP (MaskFormer 40.1 against 49.5 for Swin-HTC++, the Hybrid Task Cascade instance specialist on a Swin backbone [[Chen et al. 2019](#ref-chen2019), [Liu et al. 2021](#ref-liu2021)]). Compute: MaskFormer needed 300 epochs (full passes over the training set) where HTC++ needed 72 to do better. Memory: full-resolution mask losses meant one image per 32 GB GPU. Mask2Former's thesis was that the paradigm was right and the decoder and the recipe were wrong.

## 3. The set-prediction machinery

This section is the mathematical core shared by DETR, MaskFormer, and Mask2Former. The 2022 paper changes where the losses are evaluated (§8) but inherits this structure, so we do it properly once.

### 3.1 Matching as an assignment problem

**In plain words.** The model always returns 100 predictions in no particular order. Before the loss can score them, it has to decide which prediction is meant for which ground-truth object, and it picks the pairing with the lowest total cost. That pairing is the matching, and the Hungarian algorithm computes it. Everything after it is ordinary per-pair scoring.

Pad the ground truth with $\varnothing$ entries up to size $N$, so predictions and targets are both $N$-element sets. Write $\text{cost}(i, j)$ for the cost of pairing prediction $i$ with target $j$, where target $j$ carries a class $c^{\text{gt}}_j$ and a mask $m^{\text{gt}}_j$, and the padded entries have $c^{\text{gt}}_j = \varnothing$:

$$
\text{cost}(i,j) = \mathbb{1}\!\left[c^{\text{gt}}_j \ne \varnothing\right]\Big(\!-\lambda_{\text{cls}}\,\hat p_{i}(c^{\text{gt}}_j) + \lambda_{\text{ce}}\,\mathcal{L}_{\text{ce}}\big(m_{i}, m^{\text{gt}}_j\big) + \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}}\big(m_{i}, m^{\text{gt}}_j\big)\Big).
$$

This is the convention DETR introduced and the Mask2Former matcher implements. The class term uses the raw probability rather than the log, the mask terms are the same ones used in training ($\mathcal{L}_{\text{ce}}$ and $\mathcal{L}_{\text{dice}}$, both defined in §3.2), the weights are the training weights ($\lambda_{\text{cls}} = 2$ and $\lambda_{\text{ce}} = \lambda_{\text{dice}} = 5$, where DETR's matcher put weight 1 on the class term), and everything is gated to real segments, so pairing with a padding slot costs exactly zero. Two footnotes on that. Why raw probability? A log blows up as $\hat p \to 0$, so one confidently wrong class could dominate a whole cost row. The raw probability stays in $[0,1]$, which keeps the class term on the same footing as the mask terms. It matters only for matching, and the training loss still uses the log. And one catch. The MaskFormer paper writes the class term ungated, charging $-\hat p_i(\varnothing)$ for unmatched predictions, but the released matchers (DETR's and Mask2Former's alike) build the cost matrix over real targets only, which is the gated form above.

Two small facts make the padding trick legitimate. First, an assignment between the padded sets is a permutation $\sigma \in S_N$ (this $\sigma$ is the assignment, not the sigmoid of the notation table), a one-to-one reordering of the $N$ indices onto themselves, where $S_N$, the symmetric group, is the set of all $N!$ such reorderings. Restricted to the real targets it is exactly an injection, a one-to-one map, of ground truths into predictions, so no real segment can be dropped and no prediction can serve two targets. Second, a prediction paired with $\varnothing$ contributes zero, no matter which padding slot it received, so the minimization is genuinely over "which predictions take the real targets" and nothing else. An assignment has total cost

$$
J(\sigma) = \sum_{j=1}^{N} \text{cost}(\sigma(j),\, j),
$$

and training uses the optimal assignment $\hat\sigma = \arg\min_{\sigma} J(\sigma)$. It is solved exactly in $O(N^3)$ time, cubic in the set size. The classic solver is the Hungarian algorithm [[Kuhn 1955](#ref-kuhn1955), [Munkres 1957](#ref-munkres1957)], and the released matchers call `scipy.optimize.linear_sum_assignment`, which uses a modern equivalent, the Jonker-Volgenant algorithm, exact and cubic all the same. At $N = 100$ that is on the order of $100^3 = 10^6$ elementary operations, well under a millisecond on a CPU, and never the bottleneck. Building the cost matrix, not solving it, is the expensive part (§8).

**Aside (the Hungarian algorithm in full, skippable).** In the code the matcher is one `scipy` call, but the algorithm inside is a small classic. Its correctness is a clean duality argument, and it is where the $O(N^3)$ came from.

**As a linear program.** Encode an assignment as $x \in \{0,1\}^{N\times N}$ with $x_{ij} = 1$ when prediction $i$ takes target $j$. Relaxing the integrality to $x \ge 0$ gives a linear program,

$$
\min_{x \ge 0}\ \sum_{i,j}\text{cost}(i,j)\,x_{ij}
\quad\text{s.t.}\quad
\textstyle\sum_j x_{ij} = 1\ (\forall i),\quad \sum_i x_{ij} = 1\ (\forall j).
$$

The relaxation lets a prediction split fractionally across targets, half to one and half to another, which no real assignment can do. The surprise is that the freedom never helps. An optimal plan can always be taken to be a whole-number permutation, so solving the easy continuous problem solves the hard combinatorial one for free. That comes from the constraint matrix. Here is why.

The matrix is the incidence matrix of a bipartite graph. Its rows are the $N$ prediction constraints and the $N$ target constraints, and each variable $x_{ij}$ appears in exactly two of them, prediction $i$'s row and target $j$'s row. So the rows carry a 2-coloring, predictions against targets, with every column showing a single 1 in each color. That structure makes the matrix *totally unimodular*, every square submatrix having determinant $-1$, $0$, or $+1$.

**Lemma (total unimodularity).** *Every square submatrix $B$ of the constraint matrix has $\det B \in \{-1, 0, +1\}$.*

**Proof.** Induct on the order $k$ of $B$.

*Base* ($k = 1$): $B$ is a single entry, $0$ or $1$, so $\det B \in \{0, 1\}$.

*Step* ($k > 1$): suppose the claim holds at every order below $k$. If some column of $B$ holds fewer than two 1s, take such a column. Otherwise every column holds exactly two, the most any column can carry. That gives three cases.

- *Zero 1s.* The column is all zeros, so $\det B = 0$.
- *One 1,* at row $r$, column $c$ of $B$. Laplace-expand along that column. Only the single nonzero entry contributes, so $\det B = (-1)^{r+c}\det B'$, where $B'$ deletes row $r$ and column $c$. Now $B'$ has order $k-1$, so $\det B' \in \{-1,0,1\}$ by the hypothesis, hence $\det B = \pm\det B' \in \{-1,0,1\}$.
- *Two 1s in every column.* Every constraint-matrix column carries its two 1s in one prediction-row and one target-row, so each column of $B$ has one 1 among the prediction-rows $P$ and one among the target-rows $T$. Summing the rows in $P$ picks up exactly one 1 per column, giving the all-ones vector $\mathbf 1$, and summing the rows in $T$ gives $\mathbf 1$ the same way. So $\sum_{r\in P} B_r - \sum_{r\in T} B_r = \mathbf 0$, a nontrivial dependence among the rows, and $B$ is singular with $\det B = 0$.

Every case lands in $\{-1,0,1\}$. $\blacksquare$

Total unimodularity forces integrality. The feasible set is nonempty, since any permutation matrix sits in it, and bounded, since every $x_{ij} \in [0,1]$, so the minimum is attained at a vertex. At a vertex, enough of the $x_{ij} \ge 0$ constraints are tight to pin $x$ down. Those tight constraints hold their variables at $0$, and the surviving basic variables $x_B$ are fixed by the equality constraints. Dropping the one redundant equality (one is redundant because the prediction constraints and the target constraints each sum to the same equation $\sum_{i,j} x_{ij} = N$, so any single constraint follows from the other $2N-1$) leaves a square nonsingular submatrix $B$ with $B x_B = b$, where $b$ is a vector of ones. Cramer's rule gives $x_B = B^{-1} b$, and $\det B = \pm 1$ makes every entry an integer. With $x \ge 0$ and every row and column summing to $1$, each holds a single $1$ and the rest $0$, a permutation matrix. The relaxation is therefore exact, its optimum a genuine assignment with nothing to round. The dual attaches a potential to each constraint, $u_i$ per row and $v_j$ per column,

$$
\max_{u,v}\ \sum_i u_i + \sum_j v_j
\quad\text{s.t.}\quad
u_i + v_j \le \text{cost}(i,j)\ (\forall i,j),
$$

and dual feasibility is exactly the reduced cost $\tilde c(i,j) = \text{cost}(i,j) - u_i - v_j$ staying nonnegative everywhere.

**The optimality certificate.** Take any row potentials $u_1,\dots,u_N$ and column potentials $v_1,\dots,v_N$, and sum the reduced cost $\tilde c(i,j) = \text{cost}(i,j) - u_i - v_j$ along an assignment $\sigma$:

$$
\begin{aligned}
\sum_{j=1}^{N} \tilde c(\sigma(j),\, j)
&= \sum_{j=1}^{N}\big[\text{cost}(\sigma(j),\, j) - u_{\sigma(j)} - v_j\big] && \text{definition of } \tilde c\\[2pt]
&= \sum_{j=1}^{N}\text{cost}(\sigma(j),\, j) - \sum_{j=1}^{N} u_{\sigma(j)} - \sum_{j=1}^{N} v_j && \text{split the sum}\\[2pt]
&= J(\sigma) - \sum_{i=1}^{N} u_i - \sum_{j=1}^{N} v_j && \text{reindex the } u \text{ sum}.
\end{aligned}
$$

The reindexing is legitimate because $\sigma$ is a permutation. The map $j \mapsto \sigma(j)$ hits every row index once, so $\sum_j u_{\sigma(j)} = \sum_i u_i$. The two potential sums do not depend on $\sigma$, so the potentials shift every assignment's total by the same constant, and $\arg\min_\sigma$ is unchanged. Call the potentials feasible when $\tilde c(i,j) \ge 0$ everywhere, and an edge tight when $\tilde c(i,j) = 0$.

**Lemma (a tight assignment is optimal).** *If feasible potentials admit an assignment $\sigma$ that uses only tight edges, then $\sigma$ minimizes $J$.*

**Proof.** Let $\sigma'$ be any assignment. Rearranging the certificate identity,

$$
\begin{aligned}
J(\sigma')
&= \sum_{j=1}^{N} \tilde c(\sigma'(j),\, j) + \sum_{i=1}^{N} u_i + \sum_{j=1}^{N} v_j && \text{the identity above}\\[2pt]
&\ge \sum_{i=1}^{N} u_i + \sum_{j=1}^{N} v_j && \text{since every } \tilde c \ge 0.
\end{aligned}
$$

So $\sum_i u_i + \sum_j v_j$ lower-bounds the cost of every assignment. The tight $\sigma$ uses only edges with $\tilde c(\sigma(j),\, j) = 0$, so its first sum vanishes and $J(\sigma) = \sum_i u_i + \sum_j v_j$. That equals the lower bound, so no assignment costs less. $\blacksquare$

A full assignment sitting on tight edges is a certificate that meets that bound. The equality is complementary slackness. An optimal $x$ puts weight only where $\tilde c = 0$.

**How it runs.** The Hungarian method builds the assignment and the potentials together, holding two invariants throughout, $\tilde c \ge 0$ and a matching that uses only tight edges.

1. **Reduce.** Set $u_i = \min_j \text{cost}(i,j)$, then $v_j = \min_i \tilde c(i,j)$. This subtracts each prediction's cheapest option, then each target's, so a reduced cost now reads as regret against the best available, and a tight edge marks a pairing with zero regret. The reduction leaves $\tilde c \ge 0$ with a tight edge in every row and every column, and the column pass cannot wipe out a row's zero, because that zero sits in a column whose minimum is already $0$, so $v_j = 0$ there and the entry survives untouched.
2. **Match on tight edges.** Extend the matching over zero-reduced-cost edges by augmenting paths, as far as it will go. An augmenting path runs from a free prediction to a free target through edges that alternate unmatched and matched, and swapping which of its edges are in the matching adds exactly one pair.
3. **Perfect?** If the matching reaches size $N$ it lies entirely on tight edges, so by the lemma it is optimal. Stop.
4. **Otherwise lift the dual.** No augmenting path exists yet. Grow an alternating tree from an unmatched prediction along tight edges, reaching rows $I$ and columns $J$, and let $\delta = \min_{i\in I,\ j\notin J}\tilde c(i,j)$. Raise $u_i$ by $\delta$ on $I$ and lower $v_j$ by $\delta$ on $J$. Reduced costs stay nonnegative, a fresh tight edge opens at the frontier so the tree grows, and because the tree holds one more row than column (every column the tree reaches is matched, otherwise the path would already augment, and each matched column brings its row into the tree, so only the root row lacks a partner column) the dual objective $\sum_i u_i + \sum_j v_j$ rises by $\delta$. Repeat the lift until the tree touches an unmatched target, then augment along that path and return to step 2.

Each augmentation adds one edge, so $N$ of them finish the matching, each costing $O(N^2)$, for $O(N^3)$ overall. Termination is not luck. The dual objective strictly increases at every lift and can never exceed the optimal cost, so the primal and dual are squeezed together. Kuhn-Munkres schedules exactly these updates, and the Jonker-Volgenant routine `scipy` calls is a faster-constant refinement of the same primal-dual idea.

**Proposition (the matched loss ignores prediction order).** *Let $\mathcal{L}(\hat y) = \min_{\sigma\in S_N} J(\sigma; \hat y)$. Relabeling the predictions by any permutation $\pi$ leaves $\mathcal{L}$ unchanged.*

**Proof.** Let $\pi \in S_N$ relabel the predictions, so that position $i$ now holds the prediction originally at position $\pi(i)$. Under the relabeling, assignment $\sigma$ pairs target $j$ with the prediction originally at position $\pi(\sigma(j))$, so its cost is

$$
\sum_{j=1}^{N} \text{cost}\big(\pi(\sigma(j)),\, j\big) = J(\pi \circ \sigma) \qquad \text{(definition of } J\text{)}.
$$

Minimize over $\sigma$. As $\sigma$ ranges over $S_N$, the composite $\pi \circ \sigma$ ranges over all of $S_N$ too, because left multiplication by the fixed $\pi$ is a bijection of the group. So

$$
\min_{\sigma \in S_N} J(\pi \circ \sigma) = \min_{\tau \in S_N} J(\tau),
$$

which is $\mathcal{L}(\hat y)$ before relabeling. $\blacksquare$

Two lines, but it is the load-bearing property. The storage order of queries carries no information, so the loss must not see it, and with matching it provably does not. It also explains why matching quality sits upstream of everything else. The assignment decides which ground truth each query's gradient comes from, and a wrong pairing trains a query toward the wrong target with full confidence. Hold that thought for §8, where improving only the cost estimates is worth more than 2 AP.

One-to-one matching, as opposed to greedy nearest-target, matters for a subtler reason. Greedy lets two queries claim the same object and leaves another object orphaned. The global assignment forbids duplicate claims by construction, and that is what lets the trained model drop non-maximum suppression entirely.

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: Hungarian matching as cords untangling in segment space">
<source data-src="/assets/m2f/hungarian_matching.webm" type="video/webm">
<source data-src="/assets/m2f/hungarian_matching.mp4" type="video/mp4">
</video>
<figcaption>Fig. 3. The small gray panel at the start shows the scene being segmented, two birds and a dog. Gold shapes are the model's six predicted masks, green shapes the three ground-truth segments. They first appear in an arbitrary storage order. Then each mask shrinks to a small marker on the disc, a space where distance is matching cost. A connecting cord's length is the cost of that pairing, and it reddens as it stretches. Its three strands are the three cost terms: class, BCE, Dice. The coil on the right is the total cost of the current matching, and the Hungarian algorithm is the step that shortens it to the minimum by swapping which prediction takes which target. Predictions left unmatched become no-object, marked with a slashed ring and hanging on thin threads. Pairing with a padding slot adds nothing to the matching cost, and training touches these queries only through the classification term at its 0.1 no-object weight. At the end the storage order shuffles again while the markers hold still, because the loss depends on the set of predictions, not their order.</figcaption>
</figure>

### 3.2 The loss terms, with their gradients

The mask loss is binary cross-entropy (BCE) plus Dice,

$$
\mathcal{L}_{\text{mask}} = \lambda_{\text{ce}}\,\mathcal{L}_{\text{ce}} + \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}},
\qquad \lambda_{\text{ce}} = \lambda_{\text{dice}} = 5,
$$

and the total loss adds classification, $\mathcal{L} = \mathcal{L}_{\text{mask}} + \lambda_{\text{cls}}\,\mathcal{L}_{\text{cls}}$, where $\mathcal{L}_{\text{cls}} = -\log \hat p_i(c^{\text{gt}})$ is plain cross-entropy on the class head against the matched target, with $\lambda_{\text{cls}} = 2$ on matched queries and $0.1$ on $\varnothing$. The gradients show why these are the right terms.

**BCE.** At a point $x$ with mask logit $z_x$, prediction $m_x = \sigma(z_x)$, and target $g_x \in \{0,1\}$, the loss is

$$
\ell_{\text{ce}}(x) = -\big[g_x \log \sigma(z_x) + (1-g_x)\log\big(1-\sigma(z_x)\big)\big].
$$

The two logarithms differentiate through the sigmoid. With $\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)$,

$$
\begin{aligned}
\frac{d}{dz}\log\sigma(z) &= \frac{\sigma'(z)}{\sigma(z)} = 1 - \sigma(z) && \text{chain rule, then cancel } \sigma(z),\\[2pt]
\frac{d}{dz}\log\big(1-\sigma(z)\big) &= \frac{-\sigma'(z)}{1-\sigma(z)} = -\sigma(z) && \text{chain rule, then cancel } 1-\sigma(z).
\end{aligned}
$$

Substitute both, then expand and cancel:

$$
\begin{aligned}
\frac{\partial \ell_{\text{ce}}}{\partial z_x}
&= -\big[g_x\big(1 - \sigma(z_x)\big) - (1-g_x)\,\sigma(z_x)\big] && \text{substitute}\\[2pt]
&= -\big[g_x - g_x\sigma(z_x) - \sigma(z_x) + g_x\sigma(z_x)\big] && \text{expand}\\[2pt]
&= \sigma(z_x) - g_x && \text{cancel, then distribute the minus}.
\end{aligned}
$$

Clean, well conditioned (the gradient is bounded by 1 whatever the logit), and independent per point. The mask BCE sums this over the points, $\mathcal{L}_{\text{ce}} = \sum_x \ell_{\text{ce}}(x)$, and that per-point independence is also its weakness. The total gradient scales with the region's area, so large segments dominate the update and small ones barely register.

**Dice** [[Milletari et al. 2016](#ref-milletari2016)], in the soft form Mask2Former implements (Milletari's original squares each sum in the denominator, and the smoothing constants are dropped for clarity):

$$
\mathcal{L}_{\text{dice}}(m,g) = 1 - \frac{2\sum_x m_x g_x}{\sum_x m_x + \sum_x g_x}.
$$

Write $O = \sum_x m_x g_x$ for the overlap and $S = \sum_x m_x + \sum_x g_x$ for the size sum, so $\mathcal{L}_{\text{dice}} = 1 - 2O/S$. Differentiate with respect to the single entry $m_x$. Every other term of each sum is constant, so

$$
\frac{\partial O}{\partial m_x} = g_x, \qquad \frac{\partial S}{\partial m_x} = 1.
$$

Then

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{\text{dice}}}{\partial m_x}
&= -\,\frac{\partial}{\partial m_x}\!\left(\frac{2O}{S}\right) && \text{drop the } 1\\[4pt]
&= -\,\frac{2\,(\partial O/\partial m_x)\,S - 2O\,(\partial S/\partial m_x)}{S^2} && \text{quotient rule}\\[4pt]
&= -\,\frac{2\,g_x\,S - 2\,O}{S^2} && \text{insert the partials}.
\end{aligned}
$$

Read the fraction as a whole. The numerator $2g_xS - 2O$ is itself of order $S$, so each point's gradient scales like $1/S$, and summed over the region's roughly $S$ points the total gradient a segment receives is roughly independent of its area. The scale invariance is exact in one specific sense. Tile $k$ disjoint copies of the same prediction and target pattern, and every sum scales by $k$, leaving the loss unchanged. Dice weights a ten-pixel object and a sky-sized region equally. Its price sits one step further back, in logit space. When the prediction confidently misses the target, $m_x$ is near 0 exactly where $g_x = 1$, and the chain rule multiplies the healthy $\partial\mathcal{L}_{\text{dice}}/\partial m_x$ by $\sigma'(z_x) \approx 0$, so the gradient dies through the saturated sigmoid. BCE cancels that factor exactly (its logit gradient is $\sigma(z_x) - g_x$). Dice does not. Summing the two losses is the standard combination. BCE supplies gradient at every point, and Dice makes that gradient scale-invariant.

**What happened to focal loss?** MaskFormer used focal loss [[Lin et al. 2017](#ref-lin2017)] with weight $20$, cross-entropy scaled down on already-confident predictions so easy background points stop dominating the gradient. Mask2Former reverts to plain BCE at weight $5$. Focal loss exists to fight extreme foreground-background imbalance under dense evaluation, and §8's point sampling removes most of that imbalance at the source, so the simpler and better-conditioned loss wins.

**The $\varnothing$ down-weighting.** With $N = 100$ queries and typically 5 to 20 real segments, "no-object" outnumbers real matches by roughly 4 to 19 times. At full weight, "predict nothing" becomes the dominant gradient. The down-weighting to $0.1$ is the inexpensive fix for the same imbalance focal loss treats in dense detectors, inherited verbatim from DETR.

**Deep supervision.** The full loss is applied at every one of the 9 decoder layers, and once more on the pre-decoder predictions from $\mathbf{X}_0$. Ten supervised heads per forward pass. In most papers auxiliary losses are an optimization nicety. Here they are structural, because intermediate masks double as attention masks (§5), and "be a useful attention gate" is otherwise never optimized.

## 4. The meta-architecture

Mask2Former keeps MaskFormer's three-part skeleton exactly. A **backbone** extracts low-resolution features, either ResNet [[He et al. 2016](#ref-he2016)], a deep convolutional network whose residual connections let very deep stacks train without vanishing gradients, or Swin [[Liu et al. 2021](#ref-liu2021)], a vision Transformer that keeps attention inside shifting local windows to stay affordable at high resolution. A **pixel decoder** upsamples those features into a feature pyramid, the same content at several strides from coarse to fine, ending in per-pixel embeddings $\mathcal{E}_{\text{pixel}}$ at stride 4. Its default is MSDeformAttn, a Transformer pixel decoder built on deformable attention, the sparse-sampling scheme derived in §6.3. A **Transformer decoder** processes $N$ query vectors against the image features. Each output query $q_i$ yields a class distribution through a linear head, and a mask through a small MLP followed by a dot product with every pixel embedding:

$$
\hat p_i = \operatorname{softmax}(W_{\text{cls}}\, q_i), \qquad
m_i(x) = \sigma\big(\operatorname{MLP}(q_i)^\top\, \mathcal{E}_{\text{pixel}}(x)\big).
$$

The predicted label is the argmax $c_i = \arg\max_c \hat p_i(c)$. A query is therefore a slot that becomes one segment. One vector simultaneously determines what (the class head) and where (its inner product with the embedding field). Masks are decoded at stride 4 and bilinearly upsampled, each full-resolution pixel filled by blending its four nearest neighbors on the coarse grid. MaskFormer instantiated this skeleton with an FPN (feature pyramid network [[Lin et al. 2017b](#ref-lin2017fpn)], which merges coarse deep features with finer shallow ones into a multi-scale pyramid) pixel decoder and six standard Transformer decoder layers attending over a single stride-32 map. Every one of Mask2Former's contributions is a small, contained change inside this fixed skeleton, which is why its ablations decompose so cleanly (§11).

**Aside (the two backbone families, skippable).** Both turn the image into the low-resolution features the rest of the model consumes, and both are a black box as far as Mask2Former is concerned. A **ResNet** [[He et al. 2016](#ref-he2016)] is a stack of convolutional blocks, each computing $y = x + F(x)$ where $F$ is two or three convolutions. The identity path $+\,x$ means a block only has to learn the residual correction $F$, and gradients reach the early layers through that path undiminished, which is what let depth grow from tens of layers to past a hundred. A **Swin** Transformer [[Liu et al. 2021](#ref-liu2021)] runs self-attention instead, but to dodge the $O\big((H_lW_l)^2\big)$ cost of attending globally it confines attention to non-overlapping $M \times M$ windows, an $O(H_lW_l\,M^2)$ cost linear in the pixel count, and shifts the window grid by $M/2$ every other layer so information still crosses the seams. Either way the output is a pyramid of feature maps from stride 4 to stride 32, exactly what the pixel decoder consumes. The suffixes on the names mark size, Swin-T, S, B, L from tiny to large, and ResNet-50's 50 counts its layers.

<figure class="viz">
<svg class="m2f-arch" style="min-width: 560px" viewBox="0 0 720 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mask2Former overview: image, backbone, pixel decoder, a nine-layer Transformer decoder, and per-query class and mask heads. The decoder unfolds to show its nine masked-attention layers.">
<defs>
<style>
.lbl { font-family: Geist, ui-sans-serif, system-ui, sans-serif; fill: #171717; font-size: 13.5px; }
.sub { font-family: Geist, ui-sans-serif, system-ui, sans-serif; fill: #6b6b6b; font-size: 11px; }
.tag { font-family: Geist, ui-sans-serif, system-ui, sans-serif; font-size: 10.5px; }
.box { fill: #ffffff; stroke: #d4d4d4; stroke-width: 1; }
.flow { stroke: #9a9a9a; stroke-width: 1.3; fill: none; }
.hot { cursor: pointer; }
.hot rect.hero { transition: stroke-width 150ms ease; }
.hot:hover rect.hero { stroke-width: 2.2; }
</style>
<marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#9a9a9a"/></marker>
<marker id="ag" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#b8860b"/></marker>
<marker id="at" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#0d9488"/></marker>
<marker id="as" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#64748b"/></marker>
</defs>
<rect width="720" height="250" rx="6" fill="#fafafa"/>
<g class="m2f-collapsed">
<rect class="box" x="24" y="94" width="58" height="48" rx="4"/>
<text class="lbl" x="53" y="122" text-anchor="middle">Image</text>
<line class="flow" x1="82" y1="118" x2="108" y2="118" marker-end="url(#a)"/>
<rect class="box" x="110" y="90" width="94" height="56" rx="4"/>
<text class="lbl" x="157" y="114" text-anchor="middle">Backbone</text>
<text class="sub" x="157" y="131" text-anchor="middle">ResNet / Swin</text>
<line class="flow" x1="204" y1="118" x2="230" y2="118" marker-end="url(#a)"/>
<rect class="box" x="232" y="90" width="110" height="56" rx="4"/>
<text class="lbl" x="287" y="114" text-anchor="middle">Pixel decoder</text>
<text class="sub" x="287" y="131" text-anchor="middle">MSDeformAttn</text>
<line class="flow" x1="342" y1="118" x2="396" y2="118" marker-end="url(#a)"/>
<a href="#5-masked-attention" class="hot" data-m2f-open="1">
<rect class="hero" x="398" y="72" width="182" height="108" rx="8" fill="#0d9488" fill-opacity="0.05" stroke="#0d9488" stroke-width="1.4"/>
<text class="lbl" x="489" y="96" text-anchor="middle">Transformer decoder</text>
<text class="sub" x="489" y="113" text-anchor="middle">9 layers, coarse to fine</text>
<g fill="#b8860b"><circle cx="468" cy="137" r="4.2"/><circle cx="489" cy="137" r="4.2"/><circle cx="510" cy="137" r="4.2"/></g>
<rect x="427" y="158" width="124" height="20" rx="10" fill="#ffffff" stroke="#0d9488" stroke-width="1"/>
<text class="tag" x="489" y="172" text-anchor="middle" fill="#0d9488">unfold the 9 layers &#9656;</text>
</a>
<text class="tag" x="647" y="76" text-anchor="middle" fill="#6b6b6b">what</text>
<rect class="box" x="598" y="82" width="98" height="42" rx="4"/>
<text class="lbl" x="647" y="101" text-anchor="middle">Class head</text>
<text class="sub" x="647" y="116" text-anchor="middle">c &#8712; {1..K, &#8709;}</text>
<text class="tag" x="647" y="144" text-anchor="middle" fill="#6b6b6b">where</text>
<rect class="box" x="598" y="150" width="98" height="42" rx="4"/>
<text class="lbl" x="647" y="169" text-anchor="middle">Mask head</text>
<text class="sub" x="647" y="184" text-anchor="middle">&#963;(MLP(q)&#183;&#949;)</text>
<path class="flow" d="M 580 108 C 590 100, 590 103, 598 103" stroke="#b8860b" marker-end="url(#ag)"/>
<path class="flow" d="M 580 150 C 590 164, 590 171, 598 171" stroke="#b8860b" marker-end="url(#ag)"/>
<path d="M 287 146 L 287 214 L 647 214 L 647 192" fill="none" stroke="#64748b" stroke-width="1.3" marker-end="url(#as)"/>
<text class="tag" x="455" y="230" text-anchor="middle" fill="#64748b">&#949; per-pixel embeddings &#183; stride 4</text>
</g>
<g class="m2f-expanded" style="display: none">
<g class="hot" data-m2f-close="1">
<rect x="20" y="16" width="92" height="24" rx="5" fill="#ffffff" stroke="#d4d4d4" stroke-width="1"/>
<text class="sub" x="66" y="32" text-anchor="middle">&#8249; overview</text>
</g>
<text class="lbl" x="380" y="30" text-anchor="middle" font-size="14px">Inside the Transformer decoder</text>
<text class="sub" x="380" y="48" text-anchor="middle">one scale per layer, coarse to fine &#183; the 3-layer block repeats &#215;3</text>
<text class="tag" x="266" y="72" text-anchor="middle" fill="#0d9488">masked attention: each query reads only inside its own predicted mask</text>
<path d="M 132 104 C 132 86, 266 86, 266 102" fill="none" stroke="#0d9488" stroke-width="1.5" marker-end="url(#at)"/>
<path d="M 266 104 C 266 86, 400 86, 400 102" fill="none" stroke="#0d9488" stroke-width="1.5" marker-end="url(#at)"/>
<g fill="#b8860b"><circle cx="40" cy="128" r="4"/><circle cx="40" cy="140" r="4"/><circle cx="40" cy="152" r="4"/></g>
<text class="tag" x="40" y="172" text-anchor="middle" fill="#b8860b">N queries</text>
<line class="flow" x1="54" y1="140" x2="78" y2="140" stroke="#b8860b" marker-end="url(#ag)"/>
<rect class="box" x="80" y="104" width="104" height="76" rx="5"/>
<rect x="80" y="104" width="104" height="6" rx="2" fill="#404855"/>
<text class="tag" x="132" y="130" text-anchor="middle" fill="#0d9488">masked attn</text>
<text class="tag" x="132" y="146" text-anchor="middle" fill="#171717">self-attn</text>
<text class="tag" x="132" y="162" text-anchor="middle" fill="#171717">FFN</text>
<text class="tag" x="132" y="176" text-anchor="middle" fill="#6b6b6b">reads 1/32</text>
<rect class="box" x="214" y="104" width="104" height="76" rx="5"/>
<rect x="214" y="104" width="104" height="6" rx="2" fill="#64748b"/>
<text class="tag" x="266" y="130" text-anchor="middle" fill="#0d9488">masked attn</text>
<text class="tag" x="266" y="146" text-anchor="middle" fill="#171717">self-attn</text>
<text class="tag" x="266" y="162" text-anchor="middle" fill="#171717">FFN</text>
<text class="tag" x="266" y="176" text-anchor="middle" fill="#6b6b6b">reads 1/16</text>
<rect class="box" x="348" y="104" width="104" height="76" rx="5"/>
<rect x="348" y="104" width="104" height="6" rx="2" fill="#a8b1bd"/>
<text class="tag" x="400" y="130" text-anchor="middle" fill="#0d9488">masked attn</text>
<text class="tag" x="400" y="146" text-anchor="middle" fill="#171717">self-attn</text>
<text class="tag" x="400" y="162" text-anchor="middle" fill="#171717">FFN</text>
<text class="tag" x="400" y="176" text-anchor="middle" fill="#6b6b6b">reads 1/8</text>
<line class="flow" x1="184" y1="140" x2="214" y2="140" marker-end="url(#a)"/>
<line class="flow" x1="318" y1="140" x2="348" y2="140" marker-end="url(#a)"/>
<line class="flow" x1="452" y1="140" x2="486" y2="140" marker-end="url(#a)"/>
<text class="lbl" x="512" y="136" text-anchor="middle">&#215; 3</text>
<text class="sub" x="512" y="152" text-anchor="middle">9 layers</text>
<text class="tag" x="605" y="128" text-anchor="middle" fill="#b8860b">refined, to the heads</text>
<line class="flow" x1="544" y1="140" x2="596" y2="140" stroke="#b8860b" marker-end="url(#ag)"/>
<text class="sub" x="360" y="216" text-anchor="middle">the tab color on each block is the image scale it reads</text>
</g>
</svg>
<figcaption>Fig. 4. The pipeline, with the decoder unfolded on demand. A backbone and a pixel decoder turn the image into a feature pyramid at strides 1/32, 1/16, and 1/8, together with per-pixel embeddings &#949; at stride 4. The Transformer decoder refines N object queries over nine layers, and each finished query splits into two heads, a class for what and a mask for where, the mask formed as the sigmoid of the query's MLP output dotted with &#949;. The one idea the layers add is masked attention. A query's cross-attention is confined to its own predicted foreground, so it reads only inside the region it currently believes its object occupies, and each layer's mask, binarized at 0.5, sets where the next layer is allowed to read. Click the decoder to unfold the nine layers, a three-scale block (1/32, 1/16, 1/8) repeated three times, coarse to fine. Full treatment in <a class="section-ref" href="#5-masked-attention">&#167;5</a>.</figcaption>
</figure>

## 5. Masked attention

**In plain words.** Attention lets a query take a weighted average of the entire image, and because the background is vast, the average comes back mostly background. Masked attention forbids the query from averaging anywhere outside the region it currently believes its object occupies, and lets that belief improve layer by layer. The rest of this section is the how, the why, and the proof that it does what it claims.

### 5.1 The pathology, quantified

A standard decoder layer updates queries by cross-attention over the whole feature map:

$$
\mathbf{X}_l = \operatorname{softmax}\!\big(\mathbf{Q}_l \mathbf{K}_l^{\top}\big)\,\mathbf{V}_l + \mathbf{X}_{l-1},
\tag{1}
$$

with $\mathbf{Q}_l = f_Q(\mathbf{X}_{l-1})$, a learned linear map of the previous layer's queries, and $\mathbf{K}_l, \mathbf{V}_l$ linear projections of the features at resolution $H_l \times W_l$. Eq. 1 above omits the $1/\sqrt{d}$ temperature ($d$ is the per-head key dimension) and the multi-head split, the standard trick of running several attention operations in parallel over different slices of the channels and concatenating them. The real thing uses both, worth knowing if you reimplement from the equation alone.

Nothing restricts where a query looks, and two convergence studies of DETR had already blamed exactly this [[Gao et al. 2021](#ref-gao2021), [Sun et al. 2021](#ref-sun2021)]. Cross-attention takes hundreds of epochs to learn to localize. And the end state is no better. Even after convergence, averaged over COCO val (the validation split), only about 20 percent of attention mass lands on the ground-truth segment matched to each query's prediction, a number the Mask2Former authors measured on their own cross-attention baseline.

The mechanism deserves a short derivation, because it recurs across deep learning. Model the logits as two spikes, $n_f$ foreground locations at $\mu_f$ and $n_b$ background locations at $\mu_b$. The softmax mass on the foreground is the $n_f$ foreground weights over the total. Divide numerator and denominator by $n_f\, e^{\mu_f}$:

$$
\begin{aligned}
\frac{n_f\, e^{\mu_f}}{n_f\, e^{\mu_f} + n_b\, e^{\mu_b}}
&= \frac{1}{1 + \dfrac{n_b\, e^{\mu_b}}{n_f\, e^{\mu_f}}} && \text{divide through by } n_f\, e^{\mu_f}\\[6pt]
&= \frac{1}{1 + \dfrac{n_b}{n_f}\, e^{-(\mu_f - \mu_b)}} && \text{combine the exponentials}.
\end{aligned}
$$

Now suppose the object covers 2 percent of the image, so $n_f = 0.02\,|\Omega|$, $n_b = 0.98\,|\Omega|$, and $n_b/n_f = 0.98/0.02 = 49$. With a healthy margin $\mu_f - \mu_b = 2$ and $e^{-2} \approx 0.135$,

$$
\frac{1}{1 + 49\,e^{-2}} \approx \frac{1}{1 + 49 \cdot 0.135} = \frac{1}{7.62} \approx 0.13.
$$

Softmax never outputs zero, and the background dominates by sheer count. Thousands of individually negligible weights, integrated over a vast area, outweigh the foreground in the pooled value. This is a heuristic, since real logits are not two spikes, but it lands within about seven points of the measured 20 percent.

### 5.2 The mechanism, and why $-\infty$ specifically

The hypothesis is that a query does not need global image context at all. Local features from its own segment suffice to update it, and coordination between objects can flow through self-attention between the queries, which are $N = 100$ tokens rather than $H_l W_l \approx 10^4$. Masked attention implements this as an additive mask inside the softmax:

$$
\mathbf{X}_l = \operatorname{softmax}\!\big(\mathcal{M}_{l-1} + \mathbf{Q}_l \mathbf{K}_l^{\top}\big)\,\mathbf{V}_l + \mathbf{X}_{l-1},
\tag{2}
$$

$$
\mathcal{M}_{l-1}(x) =
\begin{cases}
0 & \text{if } M_{l-1}(x) = 1,\\[2pt]
-\infty & \text{otherwise,}
\end{cases}
\tag{3}
$$

where $M_{l-1}$ is the same query's mask from the previous layer, its logits bilinearly resized to the current attention resolution, then passed through the sigmoid and thresholded at $0.5$. $M_0$ comes from the input queries $\mathbf{X}_0$, which is only meaningful because $\mathbf{X}_0$ is learnable and directly supervised (§7).

Read the $-\infty$ literally. It is added to a location's score before the softmax, and $e^{-\infty} = 0$, so that location contributes exactly nothing to the weighted average while the surviving scores renormalize among themselves back to a total of 1. A merely large negative constant would let a sliver of weight leak through. Zeroing weights after the softmax would kill the leak but break the sum-to-one. Only the additive $-\infty$ does both jobs at once. Formally:

**Proposition (additive $-\infty$ masking is exact renormalization).** *Let $z \in \mathbb{R}^{n}$ be logits and $S \subseteq \{1,\dots,n\}$ a nonempty allowed set, with $\mathcal{M}_i = 0$ for $i\in S$ and $\mathcal{M}_i = -\infty$ otherwise. Then*

$$
\operatorname{softmax}(z + \mathcal{M})_i = \frac{e^{z_i}\,\mathbb{1}[i\in S]}{\sum_{j\in S} e^{z_j}},
$$

*which is exactly the softmax of the original logits restricted to $S$.*

**Proof.** Fix an index $i$ and write its softmax entry from the definition:

$$
\operatorname{softmax}(z + \mathcal{M})_i = \frac{e^{z_i + \mathcal{M}_i}}{\sum_{j=1}^{n} e^{z_j + \mathcal{M}_j}}.
$$

Evaluate the numerator by cases, using $e^{-\infty} = 0$:

$$
e^{z_i + \mathcal{M}_i} =
\begin{cases}
e^{z_i}\,e^{0} = e^{z_i}, & i \in S \ \ (\mathcal{M}_i = 0),\\[2pt]
e^{z_i}\,e^{-\infty} = 0, & i \notin S \ \ (\mathcal{M}_i = -\infty),
\end{cases}
$$

so in one line the numerator is $e^{z_i}\,\mathbb{1}[i \in S]$. Split the denominator at $S$:

$$
\begin{aligned}
\sum_{j=1}^{n} e^{z_j + \mathcal{M}_j}
&= \sum_{j\in S} e^{z_j}\,e^{0} + \sum_{j\notin S} e^{z_j}\,e^{-\infty} && \text{split over } S \text{ and its complement}\\[2pt]
&= \sum_{j\in S} e^{z_j} \;>\; 0 && \text{the second sum is }0,\text{ and }S\text{ nonempty makes the first positive}.
\end{aligned}
$$

Divide numerator by denominator:

$$
\operatorname{softmax}(z + \mathcal{M})_i = \frac{e^{z_i}\,\mathbb{1}[i \in S]}{\sum_{j\in S} e^{z_j}},
$$

the softmax of the original logits restricted to $S$. $\blacksquare$

Trivial, but it is the entire design distinction between Mask2Former and its neighbors. Zeroing weights after the softmax breaks normalization. The surviving weights sum to less than 1, so the update shrinks by whatever mass was discarded. Replacing attention with a plain average over the mask, which is K-Net's mask pooling, discards the learned ranking within the region. Masked attention keeps a proper, learned distribution over the foreground. The ablation prices these choices on COCO instance AP. Plain cross-attention scores 37.8, and SMCA's spatially modulated co-attention [[Gao et al. 2021](#ref-gao2021)], a predicted Gaussian bias pulling each query toward its estimated center, barely moves it to 37.9, so a soft spatial prior buys almost nothing. The gain comes from a hard constraint. Mask pooling reaches 43.1 and masked attention 43.7, 5.9 AP over none, and the renormalized, learned constraint beats uniform averaging by a further 0.6.

**Origins of the trick.** Additive $-\infty$ masking is not new. It is the mechanism of causal masking in the original Transformer decoder [[Vaswani et al. 2017](#ref-vaswani2017)], where a fixed triangular mask hides the future. Mask2Former's contribution is what generates the mask, not a fixed structural pattern but a predicted, spatial, per-query region, refined online by the network's own output. Read recursively, equations (2) and (3) define a loop in which prediction and attention bootstrap each other nine times. The mask decides where the query reads. What it reads improves the mask. The improved mask sharpens the next read.

<figure class="viz">
<video autoplay loop muted playsinline preload="auto" width="1920" height="1080" aria-label="Animation: masked attention as a stencil cutting background strands while survivors thicken">
<source src="/assets/m2f/masked_attention.webm" type="video/webm">
<source src="/assets/m2f/masked_attention.mp4" type="video/mp4">
</video>
<figcaption>Fig. 5. The gold circle is one query, and the blue-gray area is the image, with the warm patch the object the query is learning. Each thin strand is one image location's contribution to the query's attention, cool from background and warm from the object, and together they form a bundle whose total width is the attention mass, which always sums to 1. A slow pan across the image shows the many background strands outweighing the few from the object, the pathology of <a class="section-ref" href="#51-the-pathology-quantified">§5.1</a>. The plate that slides in is the query's mask from the previous layer. Its label, minus infinity, is Eq. 3. That score sends a location's softmax weight to exactly zero, so the shaded part of the plate blocks those locations. The blocked strands fall away while the survivors thicken in the same motion, so the bundle stays exactly as wide, the renormalization of Eq. 2. The stencil then tightens with each pass, the successive decoder layers sharpening the mask, and near the end it briefly seals shut and reopens, the empty-mask guard of <a class="section-ref" href="#53-stability-gradients-and-the-guard-rail">§5.3</a>. Finally the whole picture resolves into the masked-attention equation it has been illustrating.</figcaption>
</figure>

### 5.3 Stability, gradients, and the guard rail

Three properties keep the recursion stable. First, the residual path in (2) preserves the pre-read state, so one bad read from a wrong region cannot erase a query. Second, masks are re-predicted and re-supervised at every layer, so the attendable region can move. Deep supervision (§3.2) is what trains masks to be good gates, and it has to be, because the gate itself is non-differentiable. Thresholding kills gradients through $\mathcal{M}$, so the learning signal for mask quality arrives only through the per-layer auxiliary losses. Third is a detail that lives in the official code and not the paper, and without it training produces NaNs (not-a-number, the value floating-point math returns for $0/0$) within minutes. If a query's thresholded mask is empty at some scale, every logit is $-\infty$ and the softmax is $0/0$. The code detects such rows and flips them to fully unmasked, quietly falling back to plain cross-attention for that query at that layer:

```python
attn_mask = (mask_logits.sigmoid() < 0.5).flatten(2)   # True = forbidden
attn_mask[attn_mask.all(dim=-1)] = False               # the guard rail
```

### 5.4 Did it work?

The numbers are blunt. Foreground attention mass rises from about 0.20 to about 0.59, averaged over COCO val and over scales. Not to 1.0, because masks are imperfect and independently thresholded per resolution. Visualized attention for a cat query moves from a diffuse map whose maximum sits outside the cat to a map pinned on it. Layer-wise, a single masked-attention layer already outperforms nine stacked cross-attention layers on panoptic quality. Convergence follows, roughly an order of magnitude faster. Training takes 25 epochs with standard augmentation (random rescales and crops of the training images), 50 with the stronger large-scale jittering of §9 that buys a better final model, against MaskFormer's 300 and DETR's 500. And in the remove-one-component ablation, masked attention is the paper in one number. Deleting it costs 5.9 AP, 4.8 PQ, and 1.7 mIoU. The damage is largest on instance-level tasks, which fits the mechanism. "Attend only to your own segment" is most valuable where the failure mode is confusing identical-class neighbors, and least valuable where identity does not exist.

## 6. Feeding the decoder

### 6.1 The cost structure

Cross-attention cost per layer scales with the token count of the feature map. Building $\mathbf{Q}\mathbf{K}^\top$ and applying it to $\mathbf{V}$ is $O(N \cdot H_lW_l \cdot C)$, since each of the $N$ queries scores each of the $H_lW_l$ locations, and every score is a $C$-term dot product. At a $1024^2$ input, the pyramid gives $32^2 = 1024$ tokens at stride 32, $64^2 = 4096$ at stride 16, and $128^2 = 16384$ at stride 8, a factor of 16 between coarsest and finest. MaskFormer stayed at stride 32 everywhere. A distant pedestrian at that stride is a single cell, and it shows in MaskFormer's small-object numbers.

### 6.2 The schedule

Mask2Former's answer is a schedule, not a module. Feed one scale per decoder layer, coarse to fine: layer 1 sees $1/32$, layer 2 sees $1/16$, layer 3 sees $1/8$, and the three-layer pattern repeats $L = 3$ times for nine layers. Count token-layer pairs per forward pass. The round-robin touches $3 \times (1024 + 4096 + 16384) = 64{,}512$. Feeding all scales to every layer touches $9 \times 21{,}504 = 193{,}536$, exactly three times as many, and the ablation shows those extra reads gain almost nothing. Naive multi-scale reaches 44.0 AP at 247 GFLOPs (billions of floating-point operations per image, the standard compute proxy) against round-robin's 43.7 at 226, a 0.3 AP gain for tripling the high-resolution token visits. And it is the high resolution that pays, not the scale mixing. Stride 8 alone at every layer also reaches 44.0, still at 239 against the round-robin's 226. The schedule keeps essentially all of the benefit while touching high-resolution tokens in a third of the layers. Removing high-resolution features altogether costs 2.2 AP, the second-largest single component removal after masked attention.

Attention by itself treats the feature map as an unordered bag of vectors, so each scale's features must tell the layer where and which rung. Positions use DETR's 2-D sinusoidal embedding: for each axis and channel pair $i$,

$$
\begin{aligned}
\text{PE}(\text{pos}, 2i) &= \sin\!\big(\text{pos}/10000^{2i/(C/2)}\big),\\
\text{PE}(\text{pos}, 2i{+}1) &= \cos\!\big(\text{pos}/10000^{2i/(C/2)}\big),
\end{aligned}
$$

computed separately for $x$ and $y$ and concatenated. The intuition is that each position gets a unique fingerprint of phases across many frequencies, and shifting the position rotates those phases in a predictable way, which is what lets a dot product read off relative offsets. The rung is a learnable scale-level embedding $e_{\text{lvl}} \in \mathbb{R}^{1\times C}$ per resolution, borrowed from Deformable DETR [[Zhu et al. 2021](#ref-zhu2021)].

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: three resolution panes and an orb breathing coarse to fine">
<source data-src="/assets/m2f/scales_breathe.webm" type="video/webm">
<source data-src="/assets/m2f/scales_breathe.mp4" type="video/mp4">
</video>
<figcaption>Fig. 6. The three panes are the same image at the feature pyramid's three strides. The most washed-out pane is stride 32, where the small duckling disappears because the grid is too coarse to hold it. Each pane's dots are its tokens, and the count quadruples from stride 32 to 16 to 8 (a 1 to 4 to 16 ratio), so the finest pane, stride 8, has by far the most tokens and is the most expensive. The gold circle is one query, its glow pulsing each time it reads. It reads one pane per layer, coarse to fine, three times over, the round-robin schedule, and the ticks at the bottom right count those nine reads. The sagging slab is the rejected alternative, using all scales at every layer, three times the tokens for a negligible 0.3 AP, and the query moves through it slowly.</figcaption>
</figure>

### 6.3 The pixel decoder is a free variable, and that is a finding

Because masked attention only needs some pyramid, any pixel decoder plugs in. The default is a 6-layer multi-scale deformable attention Transformer (MSDeformAttn) over strides $1/8$, $1/16$, $1/32$, plus an upsample fused with the stride-4 backbone features to give the per-pixel embedding map. Deformable attention [[Zhu et al. 2021](#ref-zhu2021)] replaces dense attention with a learned sparse sample. For a query feature $z_q$ at reference point $\hat p_q$ (the point on the feature grid the query samples around, from which its learned offsets reach out),

$$
\text{MSDeformAttn}\big(z_q, \hat p_q, \{x^l\}\big) = \sum_{m=1}^{M} W_m \Big[ \sum_{l=1}^{L}\sum_{k=1}^{K} A_{mlqk}\; W'_m\, x^l\big(\psi_l(\hat p_q) + \Delta p_{mlqk}\big) \Big],
$$

where $m$ indexes heads, $l$ levels, and $k$ sampling points, with $M$, $L$, $K$ their counts (all local to this equation, so this $K$ is not the class count and this $\hat p_q$ is a point on the grid, not a class probability), $W_m$, $W'_m$ the per-head output and value projections, and $x^l$ the level-$l$ feature map. The offsets $\Delta p_{mlqk}$ and the pre-softmax attention logits are linear projections of $z_q$, and the weights $A_{mlqk}$ come from a softmax of those logits over $(l,k)$. The map $\psi_l$ rescales the reference point to level $l$, and bilinear interpolation reads the fractional positions. Cost per query is $O(MLK\,C)$, from $M \times L \times K$ sampled reads, each a $C$-wide vector, instead of touching all $\sum_l H_lW_l$ locations. A constant number of taps per query is what makes a Transformer pixel decoder affordable at stride 8.

The cross-decoder ablation restates the whole thesis in miniature. FPN scores 41.5 AP. Among classic pyramids, BiFPN, an FPN with weighted two-way cross-scale fusion, is best for instance-level tasks, and FaPN, an FPN that aligns upsampled features with finer ones before merging, is best for semantic [[Tan et al. 2020](#ref-tan2020), [Huang et al. 2021](#ref-huang2021)]. Module design re-fragments by task, which is exactly what a universal architecture is meant to end, and only MSDeformAttn wins across all three tasks at once. A universal model doubles as a testbed. A module is not better until it is better everywhere.

## 7. Rewiring the decoder layer

Three changes, zero extra FLOPs, ablated separately, jointly worth about 1.4 AP, 1.1 PQ, and 0.9 mIoU.

**Masked attention comes first.** The vanilla order is self-attention, then cross-attention, then the feed-forward network. Mask2Former swaps to masked attention, then self-attention, then FFN. At layer 1 the query features are image-independent parameters, so self-attention among them mixes priors that carry no information about this image. Read first, coordinate after. Reverting the order costs 0.5 AP.

**Query features are learnable and directly supervised.** DETR zero-initializes query features and learns only positional embeddings. Mask2Former makes $\mathbf{X}_0$ learnable and supervises the masks decoded from it before the decoder runs, which is also what makes $M_0$ a meaningful first gate. In the ablation, learnable without supervision scores the same as zero-init (42.9 AP), and with supervision 43.7. Supervision is the ingredient, not learnability. Functionally, the supervised $\mathbf{X}_0$ acts as a region-proposal network [[Ren et al. 2015](#ref-ren2015)], the front stage of a two-stage detector that emits class-agnostic candidate regions, and the decoder is an iterative proposal refiner.

**Dropout**, which randomly zeroes a fraction of activations during training as a guard against overfitting, **is removed**, for 0.7 AP. Attention maps here double as localization signals that gate the next layer's reads, and zeroing them at random injects noise into exactly the pathway the architecture depends on.

## 8. Losses: match on points, train on points

### 8.1 The memory problem and the estimator

**In plain words.** Comparing two masks at every pixel is exact but expensive. A few thousand sampled points give essentially the same answer at a fraction of the cost. What is less obvious is that choosing those points well does more than save memory. It also improves the matching itself.

Evaluating the mask losses, focal plus Dice in MaskFormer's case, densely for 100 predictions against all ground truths to build the cost matrix, then again for the matched pairs, at every supervised head, is what pinned MaskFormer at one image per 32 GB GPU. Following PointRend [[Kirillov et al. 2020](#ref-kirillov2020)], Mask2Former evaluates all mask losses on $K_{\text{pt}} = 12{,}544 = 112^2$ sampled points instead of full masks.

The statistical footing takes three lines. Let $x_1,\dots,x_{K_{\text{pt}}}$ be drawn independently and uniformly from the grid $\Omega$, and let $\ell(x)$ be a per-point loss. The estimator $\hat{\mathcal{L}} = \frac{1}{K_{\text{pt}}}\sum_{k=1}^{K_{\text{pt}}}\ell(x_k)$ has mean

$$
\begin{aligned}
\mathbb{E}\big[\hat{\mathcal{L}}\big]
&= \frac{1}{K_{\text{pt}}}\sum_{k=1}^{K_{\text{pt}}}\mathbb{E}\big[\ell(x_k)\big] && \text{linearity of expectation}\\[2pt]
&= \mathbb{E}\big[\ell(x_1)\big] && \text{the } x_k \text{ are identically distributed}\\[2pt]
&= \sum_{x\in\Omega}\frac{1}{|\Omega|}\,\ell(x) = \mathcal{L}_{\text{dense}} && \text{each } x_k \text{ uniform on } \Omega.
\end{aligned}
$$

For the variance, pull out the constant, then use independence:

$$
\begin{aligned}
\operatorname{Var}\big[\hat{\mathcal{L}}\big]
&= \frac{1}{K_{\text{pt}}^2}\operatorname{Var}\Big[\sum_{k=1}^{K_{\text{pt}}}\ell(x_k)\Big] && \operatorname{Var}[cX] = c^2\operatorname{Var}[X]\\[2pt]
&= \frac{1}{K_{\text{pt}}^2}\sum_{k=1}^{K_{\text{pt}}}\operatorname{Var}\big[\ell(x_k)\big] && \text{independence}\\[2pt]
&= \frac{1}{K_{\text{pt}}}\operatorname{Var}\big[\ell(x_1)\big] && \text{identically distributed}.
\end{aligned}
$$

The estimate is unbiased and its variance shrinks like $1/K_{\text{pt}}$, so the typical error shrinks like $1/\sqrt{K_{\text{pt}}}$. At $K_{\text{pt}} = 12{,}544$ the noise is negligible in practice, while the loss touches 12,544 of the 65,536 points a $1024^2$ crop leaves at stride 4, about a fifth of the grid the dense loss used to visit, for every prediction-target pair, at every one of the ten supervised heads.

### 8.2 Two sampling rules for two jobs

The sampling distribution differs by role, for a different reason in each case. For matching the reason is a variance argument. For the final loss it is a deliberate bias.

**Matching cost: one shared uniform set.** Every prediction-target pair in an image is scored on the same uniformly sampled points. The matcher only ever compares assignments, never absolute costs, and a pixel that lands on a hard spot like an object boundary inflates many pairs' costs together, so reusing one set of points lets that shared noise cancel out of the comparison. The matching gets more stable while every cost stays essentially unbiased. The hedge is for Dice, since the argument of §8.1 is exact for per-point losses like BCE, but Dice is a ratio of point sums, and a sampled ratio picks up a bias of order $1/K_{\text{pt}}$, negligible at 12,544 points. The variance argument behind the stability follows, and is skippable.

**Aside (why sharing cancels the shared noise, skippable).** Write the point-sampled cost of pairing prediction $i$ with target $j$ as $\hat c(i,j) = \frac{1}{K_{\text{pt}}}\sum_{k}\ell_{ij}(x_k)$, where $\ell_{ij}(x)$ is the per-point cost contribution at pixel $x$ and the points $x_1,\dots,x_{K_{\text{pt}}}$ are shared across all pairs. Sharing does not change any single entry. Each $x_k$ is still uniform on $\Omega$, so $\mathbb{E}[\hat c(i,j)]$ equals the dense cost $c(i,j)$ under shared or independent sampling alike, and each entry has the same marginal variance either way. What sharing changes is the joint distribution across entries, and that is exactly what the Hungarian algorithm reads. The algorithm never uses the absolute level of a cost. To prefer one assignment $\sigma$ over another $\sigma'$ it compares $J(\sigma) - J(\sigma') = \sum_j\big[\hat c(\sigma(j),j) - \hat c(\sigma'(j),j)\big]$, a difference of sums of entries, and the noise that can flip its sign lives in the variance of entry differences. Take the smallest case, two entries $A = \hat c(i,j)$ and $B = \hat c(i',j')$, which already exhibits the mechanism. Then

$$
\operatorname{Var}[A - B] = \operatorname{Var}[A] + \operatorname{Var}[B] - 2\operatorname{Cov}(A,B),
\qquad
\operatorname{Cov}(A,B)\big|_{\text{shared}} = \frac{1}{K_{\text{pt}}}\operatorname{Cov}_{x\sim U(\Omega)}\big(\ell_{ij}(x),\,\ell_{i'j'}(x)\big).
$$

The marginal terms $\operatorname{Var}[A]$ and $\operatorname{Var}[B]$ are fixed by the uniform marginal and are identical under both schemes. Only the cross term moves. Independent per-pair points give $\operatorname{Cov}(A,B)=0$, so the two noises add. Shared points give a per-point covariance that is positive in the mask-cost regime, since a pixel that lands on a hard region such as an object boundary inflates the loss contribution for many pairs at once, so the per-point costs are positively correlated across entries. A positive covariance subtracts off the part of the sampling noise common to $A$ and $B$ and leaves only the part that distinguishes them, so $\operatorname{Var}[A-B]$ is smaller under sharing. Lower difference variance means a smaller chance that noise flips the sign of a comparison, hence a more stable argmin, and the expected difference is unchanged, so a better decision is made about the same underlying quantity. The same term-by-term cancellation extends to a general assignment comparison, where the object is the difference of sums above rather than a single entry difference. Two qualifications. The cancellation is partial, not a common additive shift, because the per-point losses are only correlated across entries, not equal up to a per-pair constant. And the positive sign of the covariance is the operative structural condition, not a theorem. Were two entries anticorrelated at the per-point level, sharing would raise their difference variance instead.

**Final loss: per-pair importance sampling.** Each matched pair gets its own points via PointRend's procedure [[Kirillov et al. 2020](#ref-kirillov2020)], spelled out in the official config rather than the paper. Oversample $3K_{\text{pt}}$ candidates uniformly, keep the $0.75\,K_{\text{pt}}$ most uncertain, meaning mask probability closest to $0.5$, which sits near boundaries, and draw the remaining $0.25\,K_{\text{pt}}$ points uniformly. Write $q$ for the resulting non-uniform sampling distribution over $\Omega$, so the training points are drawn $x_k\sim q$ rather than uniform, and the estimator averages $\ell(x_k)$ with no importance weight $1/(|\Omega|\,q(x))$ to undo $q$. Its expectation is therefore

$$
\mathbb{E}_{q}\big[\hat{\mathcal{L}}\big] = \sum_{x\in\Omega} q(x)\,\ell(x) \;\neq\; \frac{1}{|\Omega|}\sum_{x\in\Omega}\ell(x) = \mathcal{L}_{\text{dense}},
$$

a $q$-weighted mean that overweights the boundary points $q$ favors. This is not the unbiased estimator of §8.1. The uncorrected reweighting makes it a boundary-emphasized objective by design, not a cheaper copy of the dense loss, and precision is spent where the masks actually disagree.

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: sampled points on the mask error band balancing a beam">
<source data-src="/assets/m2f/shoreline_probes.webm" type="video/webm">
<source data-src="/assets/m2f/shoreline_probes.mp4" type="video/mp4">
</video>
<figcaption>Fig. 7. Green is the ground-truth mask, gold is the model's predicted mask. Where they overlap they agree. Where they differ, the orange band marks the error, a thin strip along the boundary since two decent masks mostly agree. The balance beam checks the loss estimate. The full dense loss loads the left pan and tips it down, then the few sampled points that land on the error band load the right pan and bring it level, showing that on average the small sample weighs the same as the full computation. That is the unbiased estimate of <a class="section-ref" href="#81-the-memory-problem-and-the-estimator">§8.1</a>. The three pairs on the left show the matching rule of <a class="section-ref" href="#82-two-sampling-rules-for-two-jobs">§8.2</a>, where the same set of points is reused on every pair. The mask at the top right shows the training rule, where the points concentrate on the boundary. The lattice at the end stands in for the full sample of 112 by 112 = 12,544 points.</figcaption>
</figure>

### 8.3 The result grid, read carefully

| matching on | training loss on | AP (COCO) | PQ (COCO) | mIoU (ADE20K) | memory |
|---|---|---|---|---|---|
| masks | masks | 41.0 | 50.3 | 45.9 | 18 GB |
| masks | points | 41.0 | 50.8 | 45.9 | **6 GB** |
| points | masks | 43.1 | 51.4 | **47.3** | 18 GB |
| **points** | **points** | **43.7** | **51.9** | 47.2 | **6 GB** |

Point-sampling the training loss is the memory story, a threefold reduction at essentially no accuracy cost, confirming that even the boundary-biased sampler of §8.2 matches dense training. The unbiasedness argument of §8.1 belongs to the uniform matching cost, not this training row. Point-sampling the matching cost is an accuracy story, worth over 2 AP on top of everything else. It likely helps because dense low-resolution cost matrices are dominated by easy interior and background agreement, near-saturated terms that blur the distinctions between candidate assignments, while a shared sparse sample yields a sharper cost surface. And by the proposition of §3.1, assignment quality sits upstream of every gradient. Either way, the takeaway is that the cheap version is also the better version.

## 9. The full training recipe

The recipe is explicit enough to reproduce, including the side-by-side against MaskFormer that §11 will need.

| | MaskFormer | Mask2Former |
|---|---|---|
| optimizer | AdamW [[Loshchilov & Hutter 2019](#ref-loshchilov2019)], lr $10^{-4}$ | AdamW, lr $10^{-4}$ |
| weight decay | $10^{-4}$ | **0.05** |
| backbone lr multiplier | 0.1 (CNN backbones) | 0.1 (CNN and Transformer backbones) |
| schedule (COCO) | 300 epochs at batch 64 | **50 epochs** at batch 16, lr ×0.1 at 90% and 95% of steps |
| augmentation | standard scale and crop | **LSJ** (large-scale jittering) [[Ghiasi et al. 2021](#ref-ghiasi2021)]: scale 0.1 to 2.0, fixed $1024^2$ crop |
| mask loss | focal ($\lambda{=}20$) + dice ($\lambda{=}1$), dense | **BCE ($\lambda{=}5$) + dice ($\lambda{=}5$)** on 12,544 points |
| $\lambda_{\text{cls}}$ | 1.0 | 2.0, with 0.1 on $\varnothing$ |
| decoder | 6 layers, SA→CA→FFN, dropout 0.1, stride 32 only, zero-init queries | **9 layers, MA→SA→FFN, no dropout, strides {32,16,8}×3, learnable supervised queries** |

Three terms from that table need a gloss. AdamW is Adam, an adaptive per-parameter gradient method, with decoupled weight decay. Weight decay shrinks every weight toward zero by a small fixed fraction each step, a mild guard against overfitting, and decoupled means the shrink is applied to the weights directly instead of being folded into Adam's rescaled gradient. A batch is the number of images averaged into one gradient step, the B dimension in §14's code.

Inference on COCO follows the Mask R-CNN protocol: shorter side 800, longer side at most 1333. Queries are 100 everywhere, except 200 for the largest panoptic and instance models, trained 100 epochs. The query ablation shows 100 is best for instance and semantic, 200 helps only panoptic (52.2 against 51.9 PQ, since panoptic scenes hold more segments), and 1000 actively hurts (40.3 AP against the 43.7 that 100 queries reach). Per dataset, Cityscapes [[Cordts et al. 2016](#ref-cordts2016)] uses 90k iterations (gradient steps, counted instead of epochs) at 512×1024 crops, ADE20K [[Zhou et al. 2017](#ref-zhou2017)] uses $640^2$ crops for panoptic and instance (the semantic crop size varies by backbone, $512^2$ up to Swin-S), and Mapillary Vistas [[Neuhold et al. 2017](#ref-neuhold2017)] uses 300k iterations at $1024^2$ crops.

Post-processing is inherited from MaskFormer. Semantic output is the per-pixel argmax of class-probability-weighted masks, $\arg\max_c \sum_i \hat p_i(c)\, m_i(x)$. Panoptic works the same way, with low-confidence filtering to resolve overlaps into a partition. Instance segmentation needs the calibrated ranking AP demands (§1), so the score is

$$
s_i = \hat p_i(c_i)\cdot \frac{1}{|\{x : m_i(x) > 0.5\}|}\sum_{x:\, m_i(x)>0.5} m_i(x),
$$

class confidence times average mask confidence over the foreground, because a query can be confident about what something is while imprecise about where.

## 10. Results worth remembering

One architecture, per-task training, state of the art everywhere, for the first time:

| task / dataset | Mask2Former (Swin-L) | previous best | margin |
|---|---|---|---|
| Panoptic, COCO val | **57.8 PQ** | MaskFormer 52.7, K-Net 54.6 | +5.1 / +3.2 |
| Instance, COCO val | **50.1 AP** (36.2 boundary AP) | Swin-HTC++ 49.5 (34.1) | +0.6 (+2.1) |
| Semantic, ADE20K val | **57.7 mIoU** (Swin-L, FaPN, multi-scale inference) | [BEiT](#ref-bao2022) 57.0 | +0.7 at less than half the parameters |

Boundary AP in the table is mask AP scored only on pixels near the mask edges [[Cheng et al. 2021c](#ref-cheng2021biou)], the sharpest test of boundary precision. Multi-scale in the semantic row is a test-time trick, averaging predictions over several input resolutions, not §6's decoder schedule.

At ResNet-50 scale the story is learning efficiency. Mask2Former reaches 51.9 PQ in 50 epochs against MaskFormer's 46.5 in 300, six times fewer epochs to a higher score, and 43.7 instance AP in 50 epochs against 42.5 for a heavily tuned 400-epoch Mask R-CNN.

Three second-order results carry more information than the headlines. On COCO test-dev, the held-out split whose labels are withheld and scored by an evaluation server, large-object AP reaches 71.2 (72.1 on val), beating the COCO challenge winner's 67.7 despite the winner's extra data and ensembling (averaging the predictions of several trained models), while small-object AP is 29.1 against their 36.6. The paradigm is strong on large objects and clearly behind on small ones, a known open problem. Boundary AP rises by 2.1 over HTC++ against 0.6 overall, so the stride-4 embedding map pays exactly where mask quality is decided, at the edges. And the compute-performance frontier genuinely moves, since the lightest Mask2Former beats the heaviest MaskFormer at a quarter of the FLOPs. The throughput footnote tempers it, though. The R50 (ResNet-50) panoptic model runs at 8.6 frames per second to MaskFormer's 17.6. Multi-scale attention is not free. It is very well spent.

Generalization holds without architectural change. The same Swin-L, retrained per dataset and per task, is competitive with each domain's specialists. On Cityscapes it handles all three tasks (66.6 PQ, 43.7 AP, 83.3 mIoU), and it carries to ADE20K panoptic (48.1 PQ) and Mapillary Vistas (45.5 PQ). The point is the breadth, one decoder spanning domains that each grew a separate specialist line, not any single entry.

## 11. What actually mattered

Every row is a controlled experiment on R50, across all three tasks:

| change (removed or varied) | ΔAP | ΔPQ | ΔmIoU | takeaway |
|---|---|---|---|---|
| remove masked attention | **−5.9** | **−4.8** | −1.7 | the paper, in one number |
| remove multi-scale high-res features | −2.2 | −1.7 | −1.1 | resolution is second, and the schedule makes it affordable |
| point→mask matching cost | −2.7 | −1.1 | −1.3* | assignment quality is upstream of everything |
| remove query supervision | −0.8 | −0.7 | −1.8 | supervision, not learnability, is the ingredient |
| restore dropout | −0.7 | −0.6 | 0.0 | attention maps are localization signals, do not corrupt them |
| vanilla layer order | −0.5 | −0.3 | −0.9 | read the image before talking among queries |

*at fixed point-sampled training loss. The §8.3 grid has the full two-by-two.

There is also the decomposition every reviewer secretly wants, recipe against architecture. Retraining MaskFormer with Mask2Former's training parameters lifts it from 34.0 to 37.8 AP, so LSJ, the reweighted BCE plus Dice, and the point losses transfer to other models. Swapping in the new decoder while holding the backbone, the FPN pixel decoder, and the recipe fixed takes 37.8 to 41.5, and the MSDeformAttn default closes the rest to 43.7. That is roughly forty percent recipe, forty percent decoder, and twenty percent pixel decoder, laid out where most papers would let the headline idea absorb all the credit. Steal that habit.

## 12. Limitations, read honestly

**Universal means one architecture, not one checkpoint.** A panoptic-trained Mask2Former evaluated as an instance or semantic model trails per-task training, but the gaps are small and not one-sided. On COCO the panoptic checkpoint gives up about 2 AP on instance while matching semantic, so panoptic training nearly subsumes instance already. On ADE20K and Cityscapes the pattern flips, instance nearly matches (26.5 against 26.4, 37.3 against 37.4) and the small gaps move to semantic mIoU. The paper names the next goal, train once for everything, which is exactly what OneFormer later delivered.

**Small objects remain the weak point**, and the pyramid is under-exploited, since the round-robin schedule is an efficiency compromise, not a solution. Two softer costs round out the list: the throughput gap above, and the query-count coupling, where one hyperparameter still quietly encodes how many segments your task tends to produce.

## 13. Where it went next

Mask2Former's decoder became infrastructure. The same group extended it unchanged to video, where masks become spatio-temporal tubes [[Cheng et al. 2021b](#ref-cheng2021vis)]. **OneFormer** [[Jain et al. 2023](#ref-jain2023)] closed the gap Mask2Former left open, one jointly trained model for all three tasks, by conditioning the same skeleton on a task token. **Mask DINO** [[Li et al. 2023](#ref-li2023)] unified it with DETR-style detection, letting box and mask queries help each other. The query-as-segment abstraction became the substrate for open-vocabulary segmentation, where the fixed classifier is replaced by text embeddings, and the "predict masks, classify separately" philosophy echoes in the promptable, class-agnostic design of SAM, the Segment Anything Model that segments whatever a user's click or box points at [[Kirillov et al. 2023](#ref-kirillov2023)], even though SAM's goal differs. Mask2Former itself shipped in Detectron2 [[Wu et al. 2019](#ref-wu2019)] and in `transformers` as `facebook/mask2former-*`, and it remains a standard baseline for new segmentation papers. Leaderboards have moved since 2022. The skeleton mostly has not.

## 14. Implementation corner

The reference implementation is [`facebookresearch/Mask2Former`](https://github.com/facebookresearch/Mask2Former), and its configs reproduce every table above. The core, one decoder layer with the two details people miss, in PyTorch-flavored pseudocode:

```python
def decoder_layer(x, feats_l, pos_l, lvl_emb, query_pos, prev_mask_logits, layer):
    # x: (N,B,C) queries · feats_l: (H_l*W_l,B,C) image features at this scale
    m = F.interpolate(prev_mask_logits, size=spatial(feats_l), mode="bilinear")
    attn_mask = (m.sigmoid() < 0.5).flatten(2)      # True = forbidden
    attn_mask[attn_mask.all(dim=-1)] = False        # NaN guard (official repo)
    # bool mask => no gradient through the gate; masks learn via aux losses
    # norm1..3: LayerNorm, the per-token normalization every real Transformer
    # layer carries (left out of the §0 sketch)
    # query_pos: a learned positional embedding per query (DETR's trick),
    # added whenever queries form Q or K

    x = norm1(x + cross_attn(q=with_pos(x, query_pos),          # masked attention FIRST
                             k=with_pos(feats_l, pos_l + lvl_emb),
                             v=feats_l, attn_mask=attn_mask))   # no dropout anywhere
    x = norm2(x + self_attn(with_pos(x, query_pos)))            # queries coordinate
    x = norm3(x + ffn(x))

    h = x.transpose(0, 1)                                       # (N,B,C) -> (B,N,C) for the heads
    cls_logits  = class_head(h)                                 # (B,N,K+1)
    mask_logits = einsum("bnc,bchw->bnhw", mask_head(h), pixel_embeddings)  # stride 4
    return x, cls_logits, mask_logits    # supervised here AND gates the next layer
```

Assembly: run the heads once on the learnable $\mathbf{X}_0$ before the loop, giving both an auxiliary target and $M_0$, cycle the three scales for nine layers, and feed all ten class-mask pairs to the Hungarian-matched, point-sampled loss of §8. To feel it for yourself, reproduce one row of the ablation. Set `attn_mask=None` and watch convergence collapse.

## 15. Test yourself

**Prove that adding $-\infty$ before the softmax renormalizes over the allowed set, and say precisely what post-softmax zeroing breaks.** The §5.2 proposition: $e^{z+\mathcal{M}}$ vanishes off $S$, so the weights are the restricted softmax and still sum to 1 with the learned ranking intact. Zeroing after the softmax leaves total weight below 1, so the update shrinks by the discarded mass. Mask pooling replaces the ranking with a uniform average, and the ablation prices that ranking at 0.6 AP.

**Derive the foreground attention share for an object covering fraction $\rho$ of the image with logit margin $\Delta$, and evaluate it at $\rho = 0.02$, $\Delta = 2$.** Share $= 1/\big(1 + \frac{1-\rho}{\rho}e^{-\Delta}\big) = 1/(1+49e^{-2}) \approx 0.13$. That is the pathology of §5.1, and it sits about seven points below the measured 20 percent.

**Show the matched loss is permutation-invariant.** §3.1: relabeling predictions turns the cost of $\sigma$ into the cost of $\pi\circ\sigma$, and left multiplication is a bijection of $S_N$, so the minimum is unchanged. Corollary: the storage order of queries is provably information-free.

**Gradients do not flow through a thresholded attention mask. How do masks learn to be good gates?** They do not learn through the gate. Per-layer deep supervision trains every intermediate mask directly. The auxiliary losses are load-bearing.

**Why is the matching sample shared and uniform, while the loss sample is per-pair and boundary-biased?** Shared and uniform makes cost-matrix entries comparable, since every entry is the same functional of the same points, and unbiased, since assignment must be fair. The loss needs no fairness across pairs, so the bias is spent deliberately. Uncertainty sampling concentrates gradient where the masks disagree.

**Why did instance segmentation gain the most from masked attention and semantic the least?** The failure it fixes, attention mass leaking onto other instances and the background, is what blurs identical-class neighbors together. Semantic segmentation has no identity ambiguity and tolerates broad context.

**Your Mask2Former outputs overlapping soft masks, but panoptic requires a partition. What reconciles them?** Post-processing (§9): a per-pixel argmax over class-weighted mask scores, plus low-confidence filtering. The raw output is a set. Exclusivity is imposed afterwards.

**Why can PQ evaluation use greedy matching while training cannot?** The §1 lemma: with non-overlapping segments and IoU above 0.5, matches are provably unique, so no assignment problem exists. Training-time predictions overlap freely with soft costs, and there the Hungarian step is genuinely needed.

## 16. References

1. <a name="ref-cheng2022"></a>Cheng, Misra, Schwing, Kirillov, Girdhar. "Masked-attention Mask Transformer for Universal Image Segmentation." CVPR 2022. [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)
2. <a name="ref-cheng2021"></a>Cheng, Schwing, Kirillov. "Per-Pixel Classification is Not All You Need for Semantic Segmentation." NeurIPS 2021. [arXiv:2107.06278](https://arxiv.org/abs/2107.06278)
3. <a name="ref-carion2020"></a>Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko. "End-to-End Object Detection with Transformers." ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
4. <a name="ref-kirillov2019pan"></a>Kirillov, He, Girshick, Rother, Dollár. "Panoptic Segmentation." CVPR 2019. [arXiv:1801.00868](https://arxiv.org/abs/1801.00868)
5. <a name="ref-kirillov2020"></a>Kirillov, Wu, He, Girshick. "PointRend: Image Segmentation as Rendering." CVPR 2020. [arXiv:1912.08193](https://arxiv.org/abs/1912.08193)
6. <a name="ref-zhu2021"></a>Zhu, Su, Lu, Li, Wang, Dai. "Deformable DETR." ICLR 2021. [arXiv:2010.04159](https://arxiv.org/abs/2010.04159)
7. <a name="ref-zhang2021"></a>Zhang, Pang, Chen, Loy. "K-Net: Towards Unified Image Segmentation." NeurIPS 2021. [arXiv:2106.14855](https://arxiv.org/abs/2106.14855)
8. <a name="ref-wang2021"></a>Wang, Zhu, Adam, Yuille, Chen. "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers." CVPR 2021. [arXiv:2012.00759](https://arxiv.org/abs/2012.00759)
9. <a name="ref-he2017"></a>He, Gkioxari, Dollár, Girshick. "Mask R-CNN." ICCV 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
10. <a name="ref-long2015"></a>Long, Shelhamer, Darrell. "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
11. <a name="ref-vaswani2017"></a>Vaswani et al. "Attention Is All You Need." NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
12. <a name="ref-kuhn1955"></a>Kuhn. "The Hungarian Method for the Assignment Problem." Naval Research Logistics Quarterly, 1955.
13. <a name="ref-munkres1957"></a>Munkres. "Algorithms for the Assignment and Transportation Problems." Journal of the SIAM, 1957.
14. <a name="ref-milletari2016"></a>Milletari, Navab, Ahmadi. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." 3DV 2016. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
15. <a name="ref-lin2017"></a>Lin, Goyal, Girshick, He, Dollár. "Focal Loss for Dense Object Detection." ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
16. <a name="ref-lin2017fpn"></a>Lin, Dollár, Girshick, He, Hariharan, Belongie. "Feature Pyramid Networks for Object Detection." CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
17. <a name="ref-gao2021"></a>Gao, Zheng, Wang, Dai, Li. "Fast Convergence of DETR with Spatially Modulated Co-Attention." ICCV 2021. [arXiv:2101.07448](https://arxiv.org/abs/2101.07448)
18. <a name="ref-sun2021"></a>Sun, Cao, Yang, Kitani. "Rethinking Transformer-based Set Prediction for Object Detection." ICCV 2021. [arXiv:2011.10881](https://arxiv.org/abs/2011.10881)
19. <a name="ref-liu2021"></a>Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
20. <a name="ref-he2016"></a>He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition." CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
21. <a name="ref-chen2019"></a>Chen et al. "Hybrid Task Cascade for Instance Segmentation." CVPR 2019. [arXiv:1901.07518](https://arxiv.org/abs/1901.07518)
22. <a name="ref-ren2015"></a>Ren, He, Girshick, Sun. "Faster R-CNN." NeurIPS 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
23. <a name="ref-tan2020"></a>Tan, Pang, Le. "EfficientDet: Scalable and Efficient Object Detection." CVPR 2020. [arXiv:1911.09070](https://arxiv.org/abs/1911.09070)
24. <a name="ref-huang2021"></a>Huang, Lu, Cheng, He. "FaPN: Feature-Aligned Pyramid Network for Dense Image Prediction." ICCV 2021. [arXiv:2108.07058](https://arxiv.org/abs/2108.07058)
25. <a name="ref-loshchilov2019"></a>Loshchilov, Hutter. "Decoupled Weight Decay Regularization." ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
26. <a name="ref-ghiasi2021"></a>Ghiasi et al. "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation." CVPR 2021. [arXiv:2012.07177](https://arxiv.org/abs/2012.07177)
27. <a name="ref-chen2018"></a>Chen, Papandreou, Kokkinos, Murphy, Yuille. "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs." TPAMI 2018. [arXiv:1606.00915](https://arxiv.org/abs/1606.00915)
28. <a name="ref-zhao2017"></a>Zhao, Shi, Qi, Wang, Jia. "Pyramid Scene Parsing Network." CVPR 2017. [arXiv:1612.01105](https://arxiv.org/abs/1612.01105)
29. <a name="ref-wang2018"></a>Wang, Girshick, Gupta, He. "Non-local Neural Networks." CVPR 2018. [arXiv:1711.07971](https://arxiv.org/abs/1711.07971)
30. <a name="ref-fu2019"></a>Fu et al. "Dual Attention Network for Scene Segmentation." CVPR 2019. [arXiv:1809.02983](https://arxiv.org/abs/1809.02983)
31. <a name="ref-strudel2021"></a>Strudel, Garcia, Laptev, Schmid. "Segmenter: Transformer for Semantic Segmentation." ICCV 2021. [arXiv:2105.05633](https://arxiv.org/abs/2105.05633)
32. <a name="ref-xie2021"></a>Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)
33. <a name="ref-bao2022"></a>Bao, Dong, Piao, Wei. "BEiT: BERT Pre-Training of Image Transformers." ICLR 2022. [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
34. <a name="ref-everingham2015"></a>Everingham et al. "The PASCAL Visual Object Classes Challenge: A Retrospective." IJCV 2015.
35. <a name="ref-lin2014"></a>Lin et al. "Microsoft COCO: Common Objects in Context." ECCV 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
36. <a name="ref-cordts2016"></a>Cordts et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR 2016.
37. <a name="ref-zhou2017"></a>Zhou et al. "Scene Parsing through ADE20K." CVPR 2017.
38. <a name="ref-neuhold2017"></a>Neuhold, Ollmann, Rota Bulò, Kontschieder. "The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes." ICCV 2017.
39. <a name="ref-cheng2021vis"></a>Cheng et al. "Mask2Former for Video Instance Segmentation." 2021. [arXiv:2112.10764](https://arxiv.org/abs/2112.10764)
40. <a name="ref-jain2023"></a>Jain et al. "OneFormer: One Transformer to Rule Universal Image Segmentation." CVPR 2023. [arXiv:2211.06220](https://arxiv.org/abs/2211.06220)
41. <a name="ref-li2023"></a>Li et al. "Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation." CVPR 2023. [arXiv:2206.02777](https://arxiv.org/abs/2206.02777)
42. <a name="ref-kirillov2023"></a>Kirillov et al. "Segment Anything." ICCV 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
43. <a name="ref-wu2019"></a>Wu, Kirillov, Massa, Lo, Girshick. "Detectron2." 2019. [github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
44. <a name="ref-cheng2021biou"></a>Cheng, Girshick, Dollár, Berg, Kirillov. "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation." CVPR 2021. [arXiv:2103.16562](https://arxiv.org/abs/2103.16562)

## 17. Citation

Cited as:

> Massih, Peter. "Mask2Former, Dissected: One Transformer to Segment Them All." *Peter's Patches*, no. 1, Jul 2026. https://peteramassih.com/writing/mask2former.

Or:

```bibtex
@article{massih2026mask2former,
  title   = {Mask2Former, Dissected: One Transformer to Segment Them All},
  author  = {Massih, Peter},
  journal = {peteramassih.com},
  series  = {Peter's Patches},
  number  = {1},
  year    = {2026},
  month   = {July},
  url     = {https://peteramassih.com/writing/mask2former}
}
```

*Found an error or a sharper derivation? Tell me. A reference post earns the name by being corrected in public.*

<script>
// Some browsers hold autoplay until a play() call; the catch covers the ones
// that refuse before any interaction.
for (const v of document.querySelectorAll('video[autoplay]')) {
  v.play().catch(() => {});
}
for (const v of document.querySelectorAll('video[data-lazy]')) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      for (const s of e.target.querySelectorAll('source[data-src]')) {
        s.src = s.dataset.src;
      }
      e.target.load();
      e.target.play();
      io.disconnect();
    }
  }, { rootMargin: '400px' });
  io.observe(v);
}

// Fig. 4: click the decoder to swap the pipeline map for its nine-layer view.
// The trigger is a real <a href="#5..."> so it still navigates without JS.
for (const svg of document.querySelectorAll('svg.m2f-arch')) {
  const collapsed = svg.querySelector('.m2f-collapsed');
  const expanded = svg.querySelector('.m2f-expanded');
  const swap = (open) => {
    collapsed.style.display = open ? 'none' : '';
    expanded.style.display = open ? '' : 'none';
  };
  for (const el of svg.querySelectorAll('[data-m2f-open]')) {
    el.addEventListener('click', (e) => { e.preventDefault(); swap(true); });
  }
  for (const el of svg.querySelectorAll('[data-m2f-close]')) {
    el.addEventListener('click', (e) => { e.preventDefault(); swap(false); });
  }
}
</script>
