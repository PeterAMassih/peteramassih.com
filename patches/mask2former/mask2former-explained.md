---
title: "Mask2Former, Dissected: One Transformer to Segment Them All"
series: "Peter's Patches"
part: 1
description: "A complete, lil'log-style reference on Mask2Former (CVPR 2022): the full lineage from FCNs to set prediction, every equation derived, proofs of the properties the design relies on, the exact training recipe, and every ablation that mattered."
date: 2026-07-06
updated: 2026-07-06
author: "Peter Massih"
tags: [computer-vision, segmentation, transformers, mask2former, detr, set-prediction, paper-dissection]
math: true
readingTime: "~60 min"
hero: /assets/m2f/masked_attention.webm
---

> **TL;DR.** Mask2Former [[Cheng et al. 2022](#ref-cheng2022)] made a *single* architecture beat the best specialized models on panoptic, instance, and semantic segmentation simultaneously — 57.8 PQ and 50.1 AP on COCO, 57.7 mIoU on ADE20K — while training 6× faster and in a third of the memory of its predecessor. The mechanism is not scale; it is a rewired Transformer decoder whose cross-attention is *masked* to each query's own predicted foreground, a round-robin multi-scale feeding schedule, three zero-cost optimization changes, and a point-sampled loss. This post is written to be the reference on the paper: the lineage of every idea back to its origin, every equation derived rather than stated, short proofs of the properties the design silently relies on, the complete recipe, and the ablations ranked by what they actually bought.

This is the first entry in **Peter's Patches**, a series in the spirit of the reference posts everyone keeps a tab open to: one influential paper, rebuilt in public until nothing in it is taken on faith. Claims trace to the paper (arXiv:2112.01527v3) and its appendices; where a statement is my own commentary, or an implementation detail from the official repository rather than the paper, it is flagged as such.

**Contents.** [0. Background for newcomers](#0-the-background-you-need) · [1. The problem, formally](#1-the-segmentation-problem-formally) · [2. Origins: two paradigms](#2-origins-two-paradigms-and-where-they-came-from) · [3. Set prediction machinery](#3-the-set-prediction-machinery-in-full) · [4. The meta-architecture](#4-the-meta-architecture) · [5. Masked attention](#5-masked-attention) · [6. Multi-scale features](#6-feeding-the-decoder-high-resolution-without-the-bill) · [7. Decoder rewiring](#7-rewiring-the-decoder-layer-three-free-wins) · [8. Point-sampled losses](#8-losses-match-on-points-train-on-points) · [9. The recipe](#9-the-full-training-recipe) · [10. Results](#10-results-worth-remembering) · [11. Ablations, ranked](#11-what-actually-mattered-the-ablations-ranked) · [12. Limitations](#12-limitations-read-honestly) · [13. Lineage forward](#13-where-it-went-next) · [14. Implementation](#14-implementation-corner) · [15. Test yourself](#15-test-yourself) · [16. References](#16-references) · [17. Citation](#17-citation)

## 0. The background you need

Start here if any of this is new; skip to §1 if words like *attention*, *embedding*, and *loss* are already comfortable. The notation table below is a reference to return to, never something to memorize.

**Images, pixels, and the job.** An image is a grid of pixels. Segmentation is the family of tasks that decide which pixels belong together and what the resulting groups are — this post's §1 makes the three variants of "belong together" precise.

**What a neural network sees: feature maps.** A *backbone* network converts the image into a much smaller grid where each cell holds a vector summarizing a whole patch of the original — a *feature map*. "Stride 32" (written $1/32$) means the grid is 32× smaller per side, so a 1024×1024 image becomes a 32×32 summary — and here is a fact to hold onto: a small, distant object can simply *vanish* at that scale, because no cell is left to represent it. That one fact drives the entire design of §6.

**Embeddings and the dot product.** These summary vectors are *embeddings*: vectors whose geometry encodes meaning, arranged so that similar things point in similar directions. The dot product $u^\top v$ is the similarity meter — large when two vectors agree, near zero when they're unrelated. Keep this close, because the architecture's central trick (§4) is almost embarrassingly simple: a segment's mask is *nothing but* the dot product between one vector and every pixel's vector, squashed into $[0,1]$.

**Softmax.** Given any list of scores, softmax exponentiates them and divides by the total, producing positive weights that sum to 1 — a probability distribution over options. Two properties matter for everything that follows: (i) it **never outputs an exact zero** — every option, however unpromising, keeps a sliver of weight; and (ii) it only cares about score *differences*, renormalizing whatever survives. Property (i) causes the central pathology of this paper (§5.1); property (ii) is exactly what the fix exploits (§5.2).

**Attention, in one honest paragraph.** Attention is a soft, differentiable dictionary lookup. Each *query* asks a question; every location in the image offers a *key* ("how relevant am I to your question?") and a *value* ("here is what I contain"). Score each location by query·key, softmax the scores into weights, and return the weighted average of the values — that's the whole mechanism. When queries read from the image it's called **cross-attention**; when a set of tokens reads from each other, **self-attention**.

**Transformers and decoders.** A Transformer layer is attention plus a small per-token neural network, wrapped in a *residual connection* — "keep what you had, add what you just learned." The *decoder* in this post is a stack of such layers in which 100 learned query tokens repeatedly read from the image (cross-attention) and confer among themselves (self-attention); after nine layers, each query has *become* a description of one segment.

**Losses, gradients, training.** A *loss* is a single number measuring how wrong the current output is; training nudges every weight downhill on that number (gradient descent). The under-appreciated consequence: a network is shaped less by its wiring than by *what exactly you choose to measure* — and most of this paper's cleverness lives in the measuring (§3, §8).

**IoU.** Intersection-over-union: the overlap of two regions divided by their combined area. 1 means identical, 0 means disjoint — the standard ruler for "are these the same region?", used inside every metric in §1.

**Sets versus lists.** A list has an order; a set does not. The model's 100 guesses come out in arbitrary order, and a fair grader must not care about that order. This single innocent-sounding fact forces all of the machinery in §3.

**How to read this post.** Every heavy section opens in plain words before any symbol appears; proofs are short, boxed off, and skippable on a first pass; anything that is my interpretation rather than the paper's claim is flagged as commentary. Read it twice: once for the story, once for the math.

## Notation

| symbol | meaning |
|---|---|
| $I \in \mathbb{R}^{3\times H\times W}$ | input image; $H_l \times W_l$ is the spatial size of the feature map used at decoder layer $l$ |
| $N$ | number of object queries (100 by default; 200 for the largest panoptic/instance models) |
| $K$, $\varnothing$ | number of classes; the "no object" class appended to them |
| $(m_i, c_i)$ | prediction $i$: a soft binary mask $m_i \in [0,1]^{H\times W}$ and a class $c_i \in \{1,\dots,K\}\cup\{\varnothing\}$ |
| $\hat p_i(c)$ | predicted probability that query $i$ has class $c$ |
| $\mathbf{X}_l \in \mathbb{R}^{N\times C}$ | query features after decoder layer $l$; $\mathbf{X}_0$ are the learnable input query features |
| $\mathbf{Q}_l, \mathbf{K}_l, \mathbf{V}_l$ | attention projections; $\mathbf{K}_l,\mathbf{V}_l \in \mathbb{R}^{H_lW_l\times C}$ come from image features |
| $M_l \in \{0,1\}^{N\times H_lW_l}$ | binarized (threshold $0.5$) mask predictions of layer $l$, resized to the layer's resolution |
| $\boldsymbol{\mathcal{M}}_l$ | the additive attention mask built from $M_l$: $0$ on foreground, $-\infty$ elsewhere |
| $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{C\times\frac H4\times\frac W4}$ | per-pixel embeddings from the pixel decoder |
| $\sigma(\cdot)$ | the logistic sigmoid |
| $\lambda_{\text{ce}}, \lambda_{\text{dice}}, \lambda_{\text{cls}}$ | loss weights: $5.0$, $5.0$, $2.0$ (and $0.1$ on $\varnothing$) |
| $K_{\text{pt}}$ | number of sampled points for mask losses: $12{,}544 = 112^2$ |

## 1. The segmentation problem, formally

Image segmentation asks which pixels belong together, and the interesting part is that "belong together" admits several semantics. Fix a category set $\mathcal{C} = \mathcal{C}_{\text{th}} \sqcup \mathcal{C}_{\text{st}}$ split into countable *things* (cars, people) and amorphous *stuff* (road, sky), following the vocabulary formalized by [[Kirillov et al. 2019a](#ref-kirillov2019pan)]. The three tasks are then three output spaces over the same pixels.

**Semantic segmentation** is a map $f: \Omega \to \mathcal{C}$ on the pixel grid $\Omega$ — one label per pixel, no identities; two adjacent cars fuse into one `car` region. Its metric averages region overlap per class,

$$
\text{mIoU} \;=\; \frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}} \frac{|P_c \cap G_c|}{|P_c \cup G_c|},
$$

with $P_c, G_c$ the predicted and ground-truth pixel sets of class $c$ [[Everingham et al. 2015](#ref-everingham2015)].

**Instance segmentation** outputs a *set* of scored masks over things only, $\{(m_i, c_i, s_i)\}$, evaluated by mask AP: for each class, predictions are ranked by score $s_i$, a prediction counts as true positive when its mask IoU with an unclaimed ground truth exceeds a threshold $\tau$, and AP is the area under the resulting precision–recall curve, averaged over $\tau \in \{0.50, 0.55, \dots, 0.95\}$ COCO-style [[Lin et al. 2014](#ref-lin2014)]. Note what the metric quietly demands: *calibrated ranking*, not just good masks — a fact that will matter in post-processing (§9).

**Panoptic segmentation** [[Kirillov et al. 2019a](#ref-kirillov2019pan)] unifies both: every pixel receives a pair $(\text{class}, \text{instance id})$, identities on things, plain categories on stuff. Predicted and ground-truth segments are matched, and quality is

$$
\text{PQ} \;=\; \frac{\sum_{(p,g)\in TP}\text{IoU}(p,g)}{|TP| + \tfrac12|FP| + \tfrac12|FN|}
\;=\;
\underbrace{\frac{\sum_{(p,g)\in TP}\text{IoU}(p,g)}{|TP|}}_{\text{SQ}}
\;\times\;
\underbrace{\frac{|TP|}{|TP| + \tfrac12|FP| + \tfrac12|FN|}}_{\text{RQ}},
$$

where a match requires $\text{IoU} > 0.5$ — the factorization into segmentation quality and recognition quality is immediate algebra (multiply and divide by $|TP|$), but the *matching rule* hides a small theorem that makes PQ well-defined at all:

**Lemma (uniqueness of matches; Kirillov et al. 2019a).** *If predicted segments are mutually non-overlapping (as the panoptic format requires), each ground-truth segment $g$ can have $\text{IoU} > 0.5$ with at most one prediction.*

*Proof.* Suppose $p_1 \ne p_2$ both satisfy $\text{IoU}(p_k, g) > \tfrac12$. Since $p_k \cup g \supseteq g$, we get $|p_k \cap g| > \tfrac12|p_k \cup g| \ge \tfrac12|g|$ for both $k$. But $p_1, p_2$ are disjoint, so $p_1\cap g$ and $p_2\cap g$ are disjoint subsets of $g$, giving $|p_1\cap g| + |p_2\cap g| \le |g|$ — contradicting the sum exceeding $|g|$. $\square$

So above the $0.5$ threshold, matching is unambiguous and needs no Hungarian step — greedy is exact. Keep this in contrast with training-time matching (§3), where predictions overlap freely, costs are soft, and a genuine assignment problem appears.

![One scene, three labelings](/assets/m2f/query_becomes_segment.webm)
*Fig. 1 — the same pixels under the three semantics of grouping; the machinery that produces them will turn out to be identical (§2.2). (Scene: `query_becomes_segment`.)*

The observation that motivates the whole research program: these tasks differ **only in the semantics of grouping**, yet by 2021 each had its own architecture family, its own tricks, its own hardware optimizations — triplicated effort, and specializations that provably don't transfer, as we see next.

## 2. Origins: two paradigms and where they came from

### 2.1 Per-pixel classification (2015–)

The FCN of [[Long et al. 2015](#ref-long2015)] recast semantic segmentation as dense classification: a convolutional network ending in a $1{\times}1$ classifier head outputs $\hat y \in \mathbb{R}^{|\mathcal{C}|\times H\times W}$, trained with per-pixel cross-entropy $\;\mathcal{L} = -\frac{1}{|\Omega|}\sum_{x\in\Omega}\log \hat p_x(g_x)$. The lineage after it is a search for *context*: dilated/atrous convolutions and pyramid pooling in the DeepLab line and PSPNet [[Chen et al. 2018](#ref-chen2018); [Zhao et al. 2017](#ref-zhao2017)], then self-attention variants [[Wang et al. 2018](#ref-wang2018); [Fu et al. 2019](#ref-fu2019)], culminating in pure-Transformer per-pixel models like Segmenter and SegFormer [[Strudel et al. 2021](#ref-strudel2021); [Xie et al. 2021](#ref-xie2021)].

Why this family cannot do instances is worth stating precisely, because it is the formal reason universal architectures exist. A per-pixel classifier's output space is $\mathcal{C}^{\Omega}$ — a *fixed product of per-pixel labels*. Instance segmentation's output is a **set of variable cardinality** whose elements carry identities that are pure bookkeeping: swapping the names "car 1" and "car 2" yields the *same* answer. A function into $\mathcal{C}^\Omega$ has no variable for identity at all; bolting on "instance id" as extra channels fails because any fixed channel-to-identity assignment is arbitrary — the network would be punished for producing a correct answer in a different order. The disease has a name: the output is invariant under a symmetric group action, and the loss must be too. Per-pixel losses aren't. (Detection solved this circa 2015 with anchors-plus-NMS — impose an artificial order, then de-duplicate — which is exactly the hand-tuned machinery DETR was built to delete.)

### 2.2 Mask classification (2017–)

The alternative lineage outputs segments directly. **Mask R-CNN** [[He et al. 2017](#ref-he2017)] predicts a binary mask *per detected box* — mask classification, but tethered to boxes, which caps it at things and makes stuff awkward. **Max-DeepLab** [[Wang et al. 2021](#ref-wang2021)] first made mask prediction end-to-end with a Transformer for panoptic segmentation specifically. The general form arrived with **DETR** [[Carion et al. 2020](#ref-carion2020)]: represent each potential object as a learned *query* vector, decode all $N$ of them in parallel with a Transformer, and make the loss order-free via bipartite matching (§3). DETR was a detector, but its panoptic extension already predicted masks from queries. **MaskFormer** [[Cheng et al. 2021](#ref-cheng2021)] — same first author as Mask2Former — then showed the sharp result: a DETR-style mask classifier is not merely a way to unify tasks, it *beat per-pixel classifiers at semantic segmentation itself*, while **K-Net** [[Zhang et al. 2021](#ref-zhang2021)] pushed the set-prediction view into instance segmentation with dynamic kernels.

Formally, mask classification predicts

$$
\{(m_i, c_i)\}_{i=1}^{N}, \qquad m_i \in [0,1]^{H\times W},\quad c_i \in \{1,\dots,K\}\cup\{\varnothing\},
$$

and the three tasks fall out by *interpretation*: segments-as-categories (semantic), segments-as-things-with-identity (instance), both (panoptic). One architecture, one loss, three annotation schemes.

So by late 2021 universal architectures existed — and an awkward fact sat in the Mask2Former introduction: researchers kept building specialists anyway. Three numbers explain why. Accuracy: the best universal instance result trailed the best specialist by over 9 AP (MaskFormer 40.1 vs. Swin-HTC++ 49.5 [[Chen et al. 2019](#ref-chen2019); [Liu et al. 2021](#ref-liu2021)]). Compute: MaskFormer needed 300 epochs where HTC++ needed 72 to do better. Memory: full-resolution mask losses meant **one image per 32 GB GPU**. Mask2Former's thesis: the paradigm was right, the decoder and recipe were wrong.

## 3. The set-prediction machinery, in full

This section is the mathematical heart shared by DETR, MaskFormer, and Mask2Former; the 2022 paper modifies *where the losses are evaluated* (§8) but inherits this structure, so we do it properly once.

### 3.1 Matching as an assignment problem

**In plain words.** The model always answers in 100 guesses, handed over in no particular order — like a student submitting answers on unnumbered index cards. Before you can grade, you must first decide which card was meant for which question, and fairness demands the pairing that flatters the student most. That pairing step is the *matching*; the Hungarian algorithm finds the best one; everything afterwards is ordinary grading.

Pad the ground truth with $\varnothing$ entries to size $N$, so both prediction and truth are $N$-element sets. For a permutation $\sigma \in S_N$ define the matching cost

$$
\mathcal{C}(\sigma) \;=\; \sum_{j=1}^{N} \Big[ -\,\hat p_{\sigma(j)}(c^{gt}_j) \;+\; \mathbb{1}[c^{gt}_j \ne \varnothing]\big(\lambda_{\text{ce}}\,\mathcal{L}_{\text{ce}}(m_{\sigma(j)}, m^{gt}_j) + \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}}(m_{\sigma(j)}, m^{gt}_j)\big)\Big],
$$

i.e., the DETR convention: raw probability (not log-probability) for the class term, plus the same mask terms used in training, active only for real segments. Training uses the optimal assignment $\hat\sigma = \arg\min_{\sigma} \mathcal{C}(\sigma)$, computed exactly by the **Hungarian algorithm** — [[Kuhn 1955](#ref-kuhn1955)], polynomial-time form by [[Munkres 1957](#ref-munkres1957)], $O(N^3)$ in the Jonker–Volgenant implementations behind `scipy.optimize.linear_sum_assignment`; at $N{=}100$ the solve is microseconds and never the bottleneck (building the cost matrix is — see §8).

**Proposition (the matched loss is permutation-invariant).** *Let $\mathcal{L}(\hat y) = \min_{\sigma\in S_N} \mathcal{C}(\sigma; \hat y)$ be any matched loss. For any permutation $\pi$ of the predictions, $\mathcal{L}(\hat y \circ \pi) = \mathcal{L}(\hat y)$.*

*Proof.* $\mathcal{C}(\sigma; \hat y\circ\pi) = \sum_j \text{cost}(\hat y_{\pi(\sigma(j))}, y_j) = \mathcal{C}(\pi\circ\sigma; \hat y)$. As $\sigma$ ranges over $S_N$, so does $\pi\circ\sigma$ (left-multiplication is a bijection of the group), hence the minima over $\sigma$ coincide. $\square$

Two lines, but it is *the* load-bearing property: storage order of queries carries no information, so the loss must not see it — and with matching, provably, it doesn't. It also explains why matching quality is upstream of everything: the assignment decides which ground truth each query's gradient comes from; a wrong pairing trains a query toward the wrong target with full confidence. Hold that thought for §8, where improving the *cost estimates* alone is worth +2.7 AP.

One-to-one-ness (as opposed to greedy nearest-target) matters for a subtler reason: greedy lets two queries claim the same object and leaves another orphaned; the global assignment forbids duplicate claims *by construction*, which is what lets the trained model drop NMS entirely.

![Matching, visually](/assets/m2f/hungarian_matching.webm)
*Fig. 2 — predictions and ground truths as points in "segment space"; assignments as cords; the Hungarian step as untangling; permutation invariance as a shuffle that moves nothing. (Scene: `hungarian_matching`.)*

### 3.2 The loss terms, with their gradients

The mask loss is binary cross-entropy plus Dice, $\mathcal{L}_{\text{mask}} = \lambda_{\text{ce}}\mathcal{L}_{\text{ce}} + \lambda_{\text{dice}}\mathcal{L}_{\text{dice}}$ with $\lambda_{\text{ce}}{=}\lambda_{\text{dice}}{=}5$; the total adds classification, $\mathcal{L} = \mathcal{L}_{\text{mask}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}}$, $\lambda_{\text{cls}}{=}2$ on matched queries and $0.1$ on $\varnothing$. Why *these* terms is best seen from their gradients.

**BCE.** With mask logit $z_x$ and target $g_x \in \{0,1\}$ at point $x$, $\;\partial \mathcal{L}_{\text{ce}}/\partial z_x = \sigma(z_x) - g_x$ — the textbook result. Clean, well-conditioned, and *per-pixel independent*, which is exactly its weakness: summed over a region, total gradient scales with region **area**, so large segments dominate small ones.

**Dice** [[Milletari et al. 2016](#ref-milletari2016)], soft form with smoothing,

$$
\mathcal{L}_{\text{dice}}(m,g) = 1 - \frac{2\sum_x m_x g_x + 1}{\sum_x m_x + \sum_x g_x + 1},
\qquad
\frac{\partial \mathcal{L}_{\text{dice}}}{\partial m_x} = -\,\frac{2\big[g_x\,(\textstyle\sum m + \sum g) - \sum m g\big]}{(\sum m + \sum g)^2},
$$

by the quotient rule (drop the $+1$s for clarity). Read the denominator: every pixel's gradient is normalized by the *squared region size* — so the total gradient a segment receives is roughly independent of its area. Scale invariance made exact: tile $k$ disjoint copies of the same prediction/target pattern and every sum scales by $k$, leaving the loss unchanged. Dice therefore gives a ten-pixel duckling the same voice as the sky; its price is vanishing gradients when overlap is near zero. The sum of the two losses is the standard hedge: BCE supplies signal everywhere, Dice supplies scale fairness.

**What happened to focal loss?** MaskFormer used focal loss [[Lin et al. 2017](#ref-lin2017)] with weight $20.0$; Mask2Former reverts to plain BCE at $5.0$. The paper states the change without argument; my reading (flagged as commentary): focal loss exists to fight extreme foreground/background imbalance under *dense* evaluation, and §8's point sampling — uniform for matching, boundary-concentrated for training — removes most of that imbalance at the source, letting the simpler, better-conditioned loss win.

**The $\varnothing$ down-weighting.** With $N{=}100$ queries and typically 5–20 real segments, "no object" outnumbers real matches roughly 5–15×; at full weight, "predict nothing" is the dominant gradient. The $20\times$ down-weighting ($\lambda_\varnothing = 0.1$) is the cheap medicine for the same disease focal loss treats in dense detectors — inherited verbatim from DETR's $0.1$ on its background class.

**Deep supervision.** The full loss is applied at *every* one of the 9 decoder layers and once more on the pre-decoder predictions from $\mathbf{X}_0$ — ten supervised heads per forward pass. In most papers auxiliary losses are an optimization nicety; here they are structural, because intermediate masks moonlight as attention masks (§5) and "be a useful attention gate" is otherwise never optimized.

## 4. The meta-architecture

Mask2Former keeps MaskFormer's three-part skeleton exactly. A **backbone** (ResNet [[He et al. 2016](#ref-he2016)] or Swin [[Liu et al. 2021](#ref-liu2021)]) extracts low-resolution features. A **pixel decoder** upsamples them into a feature pyramid ending in per-pixel embeddings $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{C\times\frac H4\times\frac W4}$. A **Transformer decoder** processes $N$ query vectors against image features; each output query $q_i$ yields its class via a linear head, and its mask via a small MLP followed by a dot product against every pixel embedding:

$$
c_i \sim \text{softmax}(W_{\text{cls}}\, q_i), \qquad
m_i(x) \;=\; \sigma\big(\, \text{MLP}(q_i)^\top\, \mathcal{E}_{\text{pixel}}(x)\,\big).
$$

A query is therefore a *slot that becomes one segment*: one vector simultaneously determines **what** (class head) and **where** (its inner product with the embedding field). Masks are decoded at stride 4 and bilinearly upsampled. MaskFormer instantiated this with an FPN [[Lin et al. 2017b](#ref-lin2017fpn)] pixel decoder and six *standard* Transformer decoder layers attending over a single stride-32 map — and every one of Mask2Former's contributions is a surgical change inside this fixed skeleton, which is precisely why its ablations decompose so cleanly (§11).

## 5. Masked attention

**In plain words.** Attention lets a query take a weighted average of the *entire* image — and because the background is vast, the average comes back mostly background. Masked attention forbids the query from averaging anywhere outside the region it currently believes its object occupies, and lets that belief improve layer by layer. The rest of this section is the how, the why, and the proof that it does what it claims.

### 5.1 The pathology, quantified

A standard decoder layer updates queries by cross-attention over the whole image feature map:

$$
\mathbf{X}_l \;=\; \text{softmax}\!\big(\mathbf{Q}_l \mathbf{K}_l^{\!\top}\big)\,\mathbf{V}_l \;+\; \mathbf{X}_{l-1},
\tag{1}
$$

with $\mathbf{Q}_l = f_Q(\mathbf{X}_{l-1})$ and $\mathbf{K}_l,\mathbf{V}_l$ linear images of the features at resolution $H_l\times W_l$. (The paper writes Eq. 1 without the $1/\sqrt{d}$ temperature and the multi-head split for readability; the implementation is standard multi-head attention with both — worth knowing before you reimplement from the paper alone.)

Nothing restricts where a query looks, and two convergence studies of DETR had already indicted exactly this [[Gao et al. 2021](#ref-gao2021); [Sun et al. 2021](#ref-sun2021)]: it takes hundreds of epochs for cross-attention to *learn* to localize. Mask2Former's appendix adds the damning measurement of the end state: even after convergence, averaged over COCO `val`, only **≈ 20 % of attention mass lands on the foreground** of the segment each query predicts.

The mechanism deserves a formal sketch, because it recurs across deep learning. Suppose $n_f$ foreground locations carry logits around $\mu_f$ and $n_b$ background locations around $\mu_b$. Then the foreground share of attention is approximately

$$
\frac{n_f\, e^{\mu_f}}{n_f\, e^{\mu_f} + n_b\, e^{\mu_b}}
\;=\;
\frac{1}{1 + \frac{n_b}{n_f}\, e^{-(\mu_f - \mu_b)}}.
$$

An object occupying 2 % of the image gives $n_b/n_f = 49$; even a healthy logit margin $\mu_f - \mu_b = 2$ yields a foreground share of $1/(1+49e^{-2}) \approx 13\%$. **Softmax is never zero, and the background wins by headcount** — thousands of individually negligible weights, integrated over a vast area, dominate the pooled value. (Heuristic, since real logits aren't two spikes — but it lands within a few points of the measured 20 %.)

### 5.2 The mechanism, and why $-\infty$ specifically

The hypothesis is almost provocative: a query doesn't need global image context at all — local features from *its own segment* suffice to update it, and inter-object coordination can flow through self-attention *between queries* ($N{=}100$ tokens, not $H_lW_l \approx 10^4$). Masked attention implements it as an additive mask inside the softmax:

$$
\mathbf{X}_l = \text{softmax}\!\big(\boldsymbol{\mathcal{M}}_{l-1} + \mathbf{Q}_l \mathbf{K}_l^{\!\top}\big)\,\mathbf{V}_l + \mathbf{X}_{l-1},
\qquad
\boldsymbol{\mathcal{M}}_{l-1}(x) =
\begin{cases}
0 & M_{l-1}(x) = 1\\[2pt]
-\infty & \text{otherwise,}
\end{cases}
\tag{2, 3}
$$

where $M_{l-1}$ is the **same query's mask from the previous layer**, sigmoid-thresholded at $0.5$ and bilinearly resized to the current attention resolution; $M_0$ comes from the *input* queries $\mathbf{X}_0$, which is only meaningful because $\mathbf{X}_0$ is learnable and directly supervised (§7).

**Proposition (additive $-\infty$ masking = exact renormalization).** *Let $z \in \mathbb{R}^{n}$ be logits and $S \subseteq \{1..n\}$ the allowed set, with $\mathcal{M}_i = 0$ for $i\in S$ and $-\infty$ otherwise. Then*

$$
\text{softmax}(z + \mathcal{M})_i \;=\; \frac{e^{z_i}\,\mathbb{1}[i\in S]}{\sum_{j\in S} e^{z_j}},
$$

*i.e., exactly the softmax of the original logits restricted to $S$.*

*Proof.* $e^{z_i + \mathcal{M}_i} = e^{z_i}$ on $S$ and $0$ off it (taking $e^{-\infty}=0$); normalize. $\square$

Trivial, but it is the entire design distinction between Mask2Former and its neighbors. Zeroing weights *after* softmax breaks normalization; replacing attention with an average over the mask — K-Net's mask pooling, $\;\mathbf{X}_l \mathrel{+}= \frac{1}{|S|}\sum_{x\in S} \mathbf{V}_l(x)$ — discards the learned *ranking within* the region. Masked attention keeps a proper, learned distribution over the foreground. The ablation prices these choices on COCO instance AP: plain cross-attention 37.8, SMCA's soft Gaussian spatial prior 37.9, mask pooling 43.1, masked attention **43.7** — constraint helps enormously (+5.3 over none), and *renormalized, learned* constraint beats uniform averaging by a further +0.6.

**Origins of the trick.** Additive $-\infty$ masking is not new — it is *the* mechanism of causal masking in the original Transformer decoder [[Vaswani et al. 2017](#ref-vaswani2017)], where a fixed triangular mask hides the future. Mask2Former's contribution is what generates the mask: not a fixed structural pattern but a **predicted, spatial, per-query region, refined online by the network's own output**. Read recursively, equations (2–3) define a coarse-to-fine loop in which prediction and attention bootstrap each other nine times: the mask decides where the query reads; what it reads improves the mask; the improved mask sharpens the next read.

![Masked attention](/assets/m2f/masked_attention.webm)
*Fig. 3 — attention as light: the open drink returns mostly background; the stencil (the previous layer's mask) cuts the outside strands while the survivors thicken to conserve the beam — renormalization made visible. (Scene: `masked_attention`.)*

### 5.3 Stability, gradients, and the guard rail

Three properties keep the recursion from eating itself. First, the residual path in (2) preserves the pre-read state, so one bad read from a wrong region cannot erase a query. Second, masks are re-predicted and re-supervised at *every* layer, so the attendable region moves — deep supervision (§3.2) is what trains masks to be good gates, because the gate itself is **non-differentiable**: thresholding kills gradients through $\boldsymbol{\mathcal{M}}$, and the learning signal for mask quality arrives exclusively via the per-layer auxiliary losses. Third — an implementation detail from the official repository, absent from the paper, without which training NaNs within minutes: if a query's thresholded mask is *empty* at some scale, every logit is $-\infty$ and softmax is $0/0$; the code detects such rows and flips them to fully-unmasked, silently falling back to plain cross-attention for that query, that layer:

```python
attn_mask = (mask_logits.sigmoid() < 0.5).flatten(2)   # True = forbidden
attn_mask[attn_mask.all(dim=-1)] = False               # the guard rail
```

### 5.4 Did it work? The paper's own forensics

The appendix analysis is unusually direct. Foreground attention mass rises from **0.20 to 0.59** averaged over COCO `val` and over scales (not to 1.0 — masks are imperfect and independently thresholded per resolution). Visualized attention for a "cat" query moves from a diffuse map whose *maximum sits outside the cat* to a map pinned on it. Layer-wise, **one masked-attention layer already outperforms nine stacked cross-attention layers** (≈47.5 vs. 47.1 PQ read out per layer), and convergence follows: ~25 epochs with standard augmentation, ~50 with large-scale jittering, against MaskFormer's 300 and DETR's 500. And in the remove-one-component ablation, masked attention is the paper in one number: deleting it costs **−5.9 AP, −4.8 PQ, −1.7 mIoU** — largest on instance-level tasks, which fits the mechanism exactly: "attend only to your own segment" is most valuable where the failure mode is confusing *identical-class neighbors*, and least where identity doesn't exist (semantic).

## 6. Feeding the decoder: high resolution without the bill

### 6.1 The cost structure

Cross-attention cost per layer scales with the token count of the feature map: building $\mathbf{Q}\mathbf{K}^\top$ and applying it to $\mathbf{V}$ is $O(N \cdot H_lW_l \cdot C)$. At $1024^2$ input, the pyramid strides give $32^2{=}1024$, $64^2{=}4096$, and $128^2{=}16{,}384$ tokens for $1/32$, $1/16$, $1/8$ — a factor of 16 between coarsest and finest. MaskFormer stayed at $1/32$ everywhere; a distant pedestrian at that stride is a single cell, and it shows: MaskFormer's small-object AP$_S$ is 16.4.

### 6.2 The schedule

Mask2Former's answer is scheduling, not a module: feed **one scale per decoder layer, round-robin coarse→fine** — layer 1 sees $1/32$, layer 2 $1/16$, layer 3 $1/8$ — and stack the 3-layer pattern $L{=}3$ times for 9 layers. Count token-touches per full pass: round-robin touches $3\times(1024{+}4096{+}16384) = 64{,}512$ token-layer pairs; feeding *all* scales to *every* layer touches $9\times 21{,}504 = 193{,}536$ — exactly $3\times$ more, for what the ablation shows is **zero accuracy gain** (naïve multi-scale 44.0 AP at 247 GFLOPs vs. round-robin 43.7 at 226; single-scale $1/8$ everywhere: 44.0 at 239). Round-robin keeps essentially all of the benefit while touching high-resolution tokens in a third of the layers; removing high-resolution features altogether costs −2.2 AP — the second-largest single factor after masked attention.

So each scale's features must tell the layer *where* and *which rung*: a 2-D sinusoidal positional embedding $e_{\text{pos}}$, DETR's convention — for each axis and channel pair $i$,

$$
\text{PE}(\text{pos}, 2i) = \sin\!\big(\text{pos}/10000^{2i/(C/2)}\big),\qquad
\text{PE}(\text{pos}, 2i{+}1) = \cos\!\big(\text{pos}/10000^{2i/(C/2)}\big),
$$

computed separately for $x$ and $y$ and concatenated — plus a **learnable scale-level embedding** $e_{\text{lvl}} \in \mathbb{R}^{1\times C}$ per resolution, borrowed from Deformable DETR [[Zhu et al. 2021](#ref-zhu2021)].

![The schedule](/assets/m2f/scales_breathe.webm)
*Fig. 4 — at 1/32 a small object literally isn't representable; at 1/8 everything is, expensively; the decoder breathes coarse→fine, three times. (Scene: `scales_breathe`.)*

### 6.3 The pixel decoder is a free variable — and that's a finding

Because masked attention only needs *some* pyramid, any pixel decoder plugs in. The default is a 6-layer **multi-scale deformable attention Transformer** (MSDeformAttn) over strides $1/8, 1/16, 1/32$, plus a simple lateral-connected upsample to the stride-4 embedding map. Deformable attention [[Zhu et al. 2021](#ref-zhu2021)] replaces dense attention with a learned sparse sample: for query feature $z_q$ at reference point $\hat p_q$,

$$
\text{MSDeformAttn}(z_q, \hat p_q, \{x^l\}) = \sum_{m=1}^{M} W_m \Big[ \sum_{l=1}^{L}\sum_{k=1}^{K} A_{mlqk}\; W'_m\, x^l\big(\phi_l(\hat p_q) + \Delta p_{mlqk}\big) \Big],
$$

where the offsets $\Delta p$ and the (softmax-normalized over $l,k$) weights $A$ are linear functions of $z_q$, $\phi_l$ rescales the reference point to level $l$, and bilinear interpolation reads fractional positions. Cost per query is $O(MLK\,C)$ — a constant number of taps instead of $\sum_l H_lW_l$ — which is what makes a Transformer pixel decoder affordable at stride 8.

The cross-decoder ablation then quietly restates the paper's whole thesis in miniature: FPN scores 41.5 AP; among classic pyramids **BiFPN is best for instance-level tasks** (43.5 AP) while **FaPN is best for semantic** (46.8 mIoU) [[Tan et al. 2020](#ref-tan2020); [Huang et al. 2021](#ref-huang2021)] — module design re-fragments by task, exactly the disease being cured — and only MSDeformAttn wins across all three tasks at once (43.7 / 51.9 / 47.2). A universal model, the authors note, doubles as a *testbed*: a module isn't better until it's better everywhere.

## 7. Rewiring the decoder layer: three free wins

Zero extra FLOPs, ablated separately, worth +1.4 AP / +1.1 PQ / +0.9 mIoU jointly.

**Masked attention comes first.** Vanilla order is self-attention → cross-attention → FFN. Mask2Former swaps to **masked-attention → self-attention → FFN** on a clean argument: at layer 1 the query features are image-independent parameters, so self-attention among them mixes priors carrying no information about *this* image. Read first, coordinate after. Reverting costs −0.5 AP.

**Query features are learnable and directly supervised.** DETR zero-initializes query *features* and learns only positional embeddings. Mask2Former makes $\mathbf{X}_0$ learnable **and supervises the masks decoded from it before the decoder runs** — which is also what makes $M_0$ a meaningful first gate. The ablation is pointed: learnable-without-supervision scores *identical* to zero-init (42.9 AP); with supervision, 43.7. Supervision is the ingredient, not learnability. Functionally the supervised $\mathbf{X}_0$ is a region-proposal network [[Ren et al. 2015](#ref-ren2015)] reborn: its pre-decoder masks already reach 50.3 class-agnostic AR@100 on COCO against 57.7 after layer 9 — the decoder is an iterative proposal *refiner*.

**Dropout is removed**, for +0.7 AP. The paper reports the effect without theory; my commentary: attention maps here moonlight as localization signals gating the next layer's reads, and randomly zeroing them injects noise into precisely the pathway the architecture depends on.

## 8. Losses: match on points, train on points

### 8.1 The memory problem and the estimator

**In plain words.** Comparing two masks pixel by pixel is like weighing an entire beach to learn its average grain size: exact, and absurdly heavy. A few hundred well-chosen scoops give the same answer for a fraction of the effort — and, surprisingly, choosing the scoops well turns out to improve the *grading itself*, not just the bill.

Evaluating BCE + Dice densely for (100 predictions × all ground truths) to build the cost matrix, then again for matched pairs, across **10 supervised heads**, is what pinned MaskFormer at one image per 32 GB GPU. Following PointRend [[Kirillov et al. 2020](#ref-kirillov2020)], Mask2Former evaluates all mask losses on $K_{\text{pt}} = 12{,}544 = 112^2$ **sampled points** instead of full masks.

The statistical footing takes one line. For points $x_1,\dots,x_{K}\stackrel{iid}{\sim}\text{Unif}(\Omega)$ and per-pixel loss $\ell$,

$$
\mathbb{E}\Big[\tfrac1K\sum_{k}\ell(x_k)\Big] \;=\; \tfrac{1}{|\Omega|}\sum_{x\in\Omega}\ell(x) \;=\; \mathcal{L}_{\text{dense}},
\qquad
\text{Var} = \tfrac{1}{K}\,\text{Var}_{x}\big(\ell(x)\big),
$$

an **unbiased** Monte-Carlo estimate whose noise shrinks as $1/K$ — and at $K{=}12{,}544$ on masks decoded at stride 4, the estimate is essentially exact while touching ~5 % of the points.

### 8.2 Two sampling rules for two jobs

Where the design gets genuinely clever is that the *sampling distribution differs by role*:

**Matching cost: one shared uniform set.** All prediction/ground-truth pairs in an image are evaluated on the *same* uniformly-sampled points. Sharing is the point — every entry of the cost matrix is the same functional of the same sample, so entries are comparable and the $\arg\min$ isn't corrupted by independent sampling noise. (With per-pair samples, the assignment would sometimes be decided by luck; with a shared sample, sampling noise shifts all entries coherently.)

**Final loss: per-pair importance sampling.** Each matched pair gets its own points via PointRend's procedure — flagged as official-config detail, since the paper defers to the citation: oversample $3\times K_{\text{pt}}$ candidates uniformly, keep the $75\%$ with the most *uncertain* predictions (mask probability nearest $0.5$, i.e., near boundaries), fill the remaining $25\%$ uniformly. Note what this is **not**: there is no importance-weight correction, so the estimator is deliberately *biased* toward boundaries — a boundary-emphasized objective by design, not a cheaper copy of the dense loss. Precision is spent where the masks actually disagree.

![Point sampling](/assets/m2f/shoreline_probes.webm)
*Fig. 5 — the loss lives on a thin shoreline of disagreement; a shared constellation of probes weighs it fairly for matching; magnetized probes concentrate learning on it. (Scene: `shoreline_probes`.)*

### 8.3 The result grid, read carefully

| matching on | training loss on | AP (COCO) | PQ (COCO) | mIoU (ADE20K) | memory |
|---|---|---|---|---|---|
| masks | masks | 41.0 | 50.3 | 45.9 | 18 GB |
| masks | points | 41.0 | 50.8 | 45.9 | **6 GB** |
| points | masks | 43.1 | 51.4 | 47.3 | 18 GB |
| **points** | **points** | **43.7** | **51.9** | **47.2** | **6 GB** |

Point-sampling the *training loss* is the memory story: **3× reduction, zero accuracy cost** — the unbiasedness argument, verified. Point-sampling the *matching cost* is an **accuracy** story: **+2.1 to +2.7 AP**, on top of everything else. The paper offers no mechanism; my reading (commentary): dense low-resolution cost matrices are dominated by easy interior and background agreement — near-saturated terms that blur distinctions between candidate assignments — while a shared sparse sample yields a sharper cost surface; and by the proposition of §3.1, assignment quality is upstream of *every* gradient. Either way, the memorable form: *the cheap version is also the better version.*

## 9. The full training recipe

Papers live or die on recipes, and Mask2Former's appendix is explicit enough to reproduce — including a side-by-side against MaskFormer that §11 will need.

| | MaskFormer | Mask2Former |
|---|---|---|
| optimizer | AdamW [[Loshchilov & Hutter 2019](#ref-loshchilov2019)], lr $10^{-4}$ | AdamW, lr $10^{-4}$ |
| weight decay | $10^{-4}$ | **0.05** |
| backbone lr multiplier | 0.1 | 0.1 (applied to Transformer backbones too, for semantic) |
| schedule (COCO) | 300 epochs | **50 epochs**, step ×0.1 at 90 % and 95 % of steps |
| batch | 16 | 16 |
| augmentation | standard scale + crop | **LSJ** [[Ghiasi et al. 2021](#ref-ghiasi2021)]: scale 0.1–2.0, fixed 1024² crop |
| mask loss | focal ($\lambda{=}20$) + dice ($\lambda{=}1$), dense | **BCE ($\lambda{=}5$) + dice ($\lambda{=}5$)** on 12,544 points |
| $\lambda_{\text{cls}}$ | 1.0 | 2.0 (0.1 on $\varnothing$) |
| decoder | 6 layers, SA→CA→FFN, dropout 0.1, $\{1/32\}$ only, zero-init queries | **9 layers, MA→SA→FFN, no dropout, $\{1/32,1/16,1/8\}{\times}3$, learnable supervised queries** |

Inference on COCO follows the Mask R-CNN protocol (shorter side 800, longer ≤ 1333). Queries: 100 everywhere except **200 for Swin-L panoptic/instance** (trained 100 epochs); the query ablation shows 100 optimal for instance and semantic, 200 better only for panoptic (52.2 vs. 51.9 PQ — panoptic scenes simply hold more segments), and 1000 actively harmful (40.3 AP). Per-dataset settings: Cityscapes [[Cordts et al. 2016](#ref-cordts2016)] 90k iterations at 512×1024 crops, whole-image inference; ADE20K [[Zhou et al. 2017](#ref-zhou2017)] panoptic/instance at 640² crops; Mapillary Vistas [[Neuhold et al. 2017](#ref-neuhold2017)] 300k iterations, poly schedule, 1024² crops, inference at longer side 2048.

Post-processing is inherited from MaskFormer: semantic output as the per-pixel argmax of class-probability-weighted masks, $\arg\max_c \sum_i \hat p_i(c)\, m_i(x)$; panoptic likewise with low-confidence filtering to resolve overlaps into a partition. Instance segmentation needs the ranking score AP demands (§1): $\;s_i = \hat p_i(c_i)\cdot \frac{1}{|m_i{>}0.5|}\sum_{x: m_i(x)>0.5} m_i(x)$ — class confidence times mask confidence, because a query can be sure *what* something is while sloppy about *where*.

## 10. Results worth remembering

One architecture, per-task training, state of the art everywhere — for the first time:

| task / dataset | Mask2Former (Swin-L) | previous best | margin |
|---|---|---|---|
| Panoptic, COCO `val` | **57.8 PQ** | MaskFormer 52.7 / K-Net 54.6 | +5.1 / +3.2 |
| Panoptic, COCO `test-dev` | **58.3 PQ** | Megvii challenge winner 54.7, with extra data + ensembles | +3.6, none of either |
| Instance, COCO `val` | **50.1 AP** (36.2 boundary AP) | Swin-HTC++ 49.5 (34.1) | +0.6 (+2.1) |
| Semantic, ADE20K `val` | **57.7 mIoU** (Swin-L-FaPN) | BEiT-L 57.0 [[Bao et al. 2022](#ref-bao2022)] | +0.7 at <½ the parameters (217M vs 502M) |

At ResNet-50 scale the story is learning efficiency: 51.9 PQ in 50 epochs against MaskFormer's 46.5 in 300 (6× faster to a better place), and 43.7 instance AP in 50 epochs against 42.5 for a heavily tuned, 400-epoch Mask R-CNN at comparable parameters.

Three second-order results carry more information than the headline. On `test-dev` instance segmentation, **AP$_L$ = 71.2, beating the challenge winner's 67.7** despite the winner's extra data and ensembling — while **AP$_S$ = 29.1 against their 36.6**: the paradigm is spectacular on large objects and clearly behind on small ones, the paper's own declared open problem. Boundary AP rises +2.1 over HTC++ against +0.6 overall — the stride-4 embedding map pays exactly at mask *edges*. And the compute-performance frontier genuinely moves — the lightest Mask2Former beats the heaviest MaskFormer at a quarter of the FLOPs — though honesty requires the throughput footnote: the R50 panoptic model runs 8.6 fps to MaskFormer's 17.6. Multi-scale attention isn't free; it is very well spent.

Generalization holds without architectural change: Cityscapes 66.6 PQ / 43.7 AP / 84.3 mIoU (the Swin-B semantic model's 84.5 edges SegFormer's 84.0), ADE20K panoptic 48.1 PQ, Mapillary 45.5 PQ / 64.7 mIoU — competitive with street-scene specialists on their home turf.

## 11. What actually mattered (the ablations, ranked)

Every row is a controlled experiment on R50, across all three tasks:

| change (removed / varied) | Δ AP | Δ PQ | Δ mIoU | takeaway |
|---|---|---|---|---|
| − masked attention | **−5.9** | **−4.8** | −1.7 | the paper, in one number |
| − multi-scale high-res features | −2.2 | −1.7 | −1.1 | resolution is #2; scheduling makes it affordable |
| mask→point matching cost | −2.1…−2.7 | −0.5 | −2.4* | the sleeper: assignment quality is upstream of everything |
| − query supervision (learnable $\mathbf{X}_0$) | −0.8 | −0.7 | −1.8 | supervision, not learnability, is the ingredient |
| + dropout back | −0.7 | −0.6 | 0.0 | attention maps are localization signals; don't corrupt them |
| cross-attention first (vanilla order) | −0.5 | −0.3 | −0.9 | read the image before talking among queries |

*at fixed point-sampled training loss; see the §8.3 grid for the full 2×2.

The appendix also runs the decomposition every reviewer secretly wants — **recipe vs. architecture**. MaskFormer retrained with Mask2Former's training parameters: 34.0 → 37.8 AP (+3.8 from recipe alone; LSJ, the reweighted BCE+dice, and point losses transfer to other models). Swapping in the new decoder while holding backbone, FPN, and recipe fixed: 37.8 → 41.5 (+3.7 from the decoder alone). The MSDeformAttn default closes it to 43.7. Roughly a third recipe, a third decoder, a third pixel decoder — stated where most papers would let the headline idea absorb all the credit. Steal that habit.

## 12. Limitations, read honestly

**"Universal" means one architecture, not one checkpoint.** A panoptic-trained Mask2Former evaluated as an instance or semantic model trails per-task training — but the gaps are small and not one-sided: on COCO, 41.7 vs. 43.7 AP (real gap) yet 61.7 vs. 61.5 mIoU (the panoptic model *wins* semantic); ADE20K 26.5 vs. 26.4 AP (tie), 46.1 vs. 47.2 mIoU; Cityscapes 37.3 vs. 37.4 AP (tie), 77.5 vs. 79.4 mIoU. Panoptic training nearly subsumes instance already; the residual gaps sit in stuff classes. The paper names the next goal — train once for everything — which is exactly what OneFormer later delivered.

**Small objects remain the weak flank** (AP$_S$ 29.9 vs. 31.0 for HTC++ on `val`), and the authors concede the pyramid is under-exploited: round-robin is an efficiency compromise, not a solution. Add two soft costs from the tables: small-scale throughput (above), and the query-count/task coupling — one hyperparameter still quietly encodes "how many segments does your task produce."

## 13. Where it went next

Mask2Former's decoder became infrastructure. The same group extended it unchanged to video (masks become spatio-temporal tubes) [[Cheng et al. 2021b](#ref-cheng2021vis)]. **OneFormer** [[Jain et al. 2023](#ref-jain2023)] closed the paper's own declared gap — one *jointly trained* model for all three tasks — by conditioning the same skeleton on a task token. **Mask DINO** [[Li et al. 2023](#ref-li2023)] unified it with DETR-style detection, letting box and mask queries help each other. The query-as-segment abstraction then became the substrate for open-vocabulary segmentation (replace the fixed classifier with text embeddings), and the "predict masks, classify separately" philosophy echoes in SAM's promptable, class-agnostic design [[Kirillov et al. 2023](#ref-kirillov2023)] even though SAM's goal differs. Mask2Former itself shipped: Detectron2 [[Wu et al. 2019](#ref-wu2019)] and `transformers` (`facebook/mask2former-*`), and it remains the baseline every new segmentation paper must beat. Leaderboards have moved since 2022; the skeleton mostly hasn't.

## 14. Implementation corner

The reference implementation is [`facebookresearch/Mask2Former`](https://github.com/facebookresearch/Mask2Former); configs reproduce every table above. The heart — one decoder layer, both details people miss included — in PyTorch-flavored pseudocode:

```python
def decoder_layer(x, feats_l, pos_l, lvl_emb, query_pos, prev_mask_logits, layer):
    # x: (N,B,C) queries · feats_l: (H_l*W_l,B,C) image features at this layer's scale
    m = F.interpolate(prev_mask_logits, size=spatial(feats_l), mode="bilinear")
    attn_mask = (m.sigmoid() < 0.5).flatten(2)      # True = forbidden
    attn_mask[attn_mask.all(dim=-1)] = False        # NaN guard (official repo)
    # bool mask => no gradient through the gate; masks learn via aux losses (§3.2)

    x = norm1(x + cross_attn(q=with_pos(x, query_pos),          # masked attention FIRST
                             k=with_pos(feats_l, pos_l + lvl_emb),
                             v=feats_l, attn_mask=attn_mask))   # no dropout anywhere
    x = norm2(x + self_attn(with_pos(x, query_pos)))            # queries coordinate
    x = norm3(x + ffn(x))

    cls_logits  = class_head(x)                                  # (B,N,K+1)
    mask_logits = einsum("bnc,bchw->bnhw", mask_head(x), pixel_embeddings)  # stride 4
    return x, cls_logits, mask_logits    # supervised here AND gates the next layer
```

Assembly: run the heads once on the learnable $\mathbf{X}_0$ before the loop (auxiliary target *and* $M_0$), cycle scales for nine layers, feed all ten (class, mask) pairs to the Hungarian-matched, point-sampled loss of §8. To *feel* the paper, reproduce one row of Table 4c: set `attn_mask=None` and watch convergence collapse.

## 15. Test yourself

**Prove that adding $-\infty$ before the softmax renormalizes over the allowed set — and say precisely what post-softmax zeroing breaks.** §5.2's proposition: $e^{z+\mathcal{M}}$ vanishes off $S$, so weights are the restricted softmax, still summing to 1 with the learned ranking intact. Zeroing after softmax leaves total weight $<1$ (the update shrinks by the discarded mass) and mask pooling replaces the ranking with a uniform average — the 43.7 vs. 43.1 ablation prices the ranking at +0.6 AP.

**Derive the foreground attention share for an object covering fraction $\rho$ of locations with logit margin $\Delta$, and evaluate at $\rho{=}0.02$, $\Delta{=}2$.** Share $= 1/(1 + \frac{1-\rho}{\rho}e^{-\Delta}) = 1/(1+49e^{-2}) \approx 13\%$ — the headcount pathology of §5.1, matching the measured ~20 %.

**Show the matched loss is permutation-invariant.** §3.1: permuting predictions permutes assignments by left-composition, a bijection of $S_N$, so the minimum is unchanged. Corollary: storage order of queries is provably information-free.

**Gradients don't flow through a thresholded attention mask — so how do masks learn to be good gates?** They don't learn *through* the gate; per-layer deep supervision trains every intermediate mask directly. The auxiliary losses are load-bearing.

**Why is the matching sample shared and uniform, but the loss sample per-pair and boundary-biased?** Shared+uniform makes cost-matrix entries comparable (same functional, same points) and unbiased — assignment must be *fair*. The loss needs no fairness across pairs, so bias is spent deliberately: PointRend's uncertainty sampling concentrates gradient where masks disagree.

**Why did instance segmentation gain most from masked attention and semantic least?** The failure it fixes — attention mass leaking onto other instances and background — is what blurs identical-class neighbors together; semantic segmentation has no identity ambiguity and tolerates broad context.

**Your Mask2Former outputs overlapping soft masks; panoptic requires a partition. What reconciles them?** Post-processing (§9): per-pixel argmax over class-weighted mask scores plus low-confidence filtering. The raw output is a set; exclusivity is imposed after.

**Why can PQ evaluation use greedy matching while training cannot?** The §1 lemma: with non-overlapping segments and IoU > 0.5, matches are provably unique — no assignment problem exists. Training-time predictions overlap freely with soft costs, so the Hungarian step is genuinely needed.

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
13. <a name="ref-munkres1957"></a>Munkres. "Algorithms for the Assignment and Transportation Problems." J. SIAM, 1957.
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
27. <a name="ref-chen2018"></a>Chen, Papandreou, Kokkinos, Murphy, Yuille. "DeepLab: Semantic Image Segmentation with Atrous Convolution and CRFs." TPAMI 2018.
28. <a name="ref-zhao2017"></a>Zhao, Shi, Qi, Wang, Jia. "Pyramid Scene Parsing Network." CVPR 2017. [arXiv:1612.01105](https://arxiv.org/abs/1612.01105)
29. <a name="ref-wang2018"></a>Wang, Girshick, Gupta, He. "Non-local Neural Networks." CVPR 2018. [arXiv:1711.07971](https://arxiv.org/abs/1711.07971)
30. <a name="ref-fu2019"></a>Fu et al. "Dual Attention Network for Scene Segmentation." CVPR 2019. [arXiv:1809.02983](https://arxiv.org/abs/1809.02983)
31. <a name="ref-strudel2021"></a>Strudel, Garcia, Laptev, Schmid. "Segmenter: Transformer for Semantic Segmentation." ICCV 2021. [arXiv:2105.05633](https://arxiv.org/abs/2105.05633)
32. <a name="ref-xie2021"></a>Xie et al. "SegFormer." NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)
33. <a name="ref-bao2022"></a>Bao, Dong, Piao, Wei. "BEiT: BERT Pre-Training of Image Transformers." ICLR 2022. [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
34. <a name="ref-everingham2015"></a>Everingham et al. "The PASCAL VOC Challenge: A Retrospective." IJCV 2015.
35. <a name="ref-lin2014"></a>Lin et al. "Microsoft COCO: Common Objects in Context." ECCV 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
36. <a name="ref-cordts2016"></a>Cordts et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR 2016.
37. <a name="ref-zhou2017"></a>Zhou et al. "Scene Parsing through ADE20K." CVPR 2017.
38. <a name="ref-neuhold2017"></a>Neuhold, Ollmann, Rota Bulò, Kontschieder. "The Mapillary Vistas Dataset." ICCV 2017.
39. <a name="ref-cheng2021vis"></a>Cheng et al. "Mask2Former for Video Instance Segmentation." 2021. [arXiv:2112.10764](https://arxiv.org/abs/2112.10764)
40. <a name="ref-jain2023"></a>Jain et al. "OneFormer: One Transformer to Rule Universal Image Segmentation." CVPR 2023. [arXiv:2211.06220](https://arxiv.org/abs/2211.06220)
41. <a name="ref-li2023"></a>Li et al. "Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation." CVPR 2023. [arXiv:2206.02777](https://arxiv.org/abs/2206.02777)
42. <a name="ref-kirillov2023"></a>Kirillov et al. "Segment Anything." ICCV 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
43. <a name="ref-wu2019"></a>Wu, Kirillov, Massa, Lo, Girshick. "Detectron2." 2019. [github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

## 17. Citation

Cited as:

> Massih, Peter. "Mask2Former, Dissected: One Transformer to Segment Them All." *Peter's Patches*, no. 1, Jul 2026. https://peteramassih.com/patches/mask2former.

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
  url     = {https://peteramassih.com/patches/mask2former}
}
```

*Found an error or a sharper derivation? Tell me — a reference post earns the name by being corrected in public.*
