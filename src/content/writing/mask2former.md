---
title: "Mask2Former, Dissected: One Transformer to Segment Them All"
description: "A complete reference on Mask2Former (CVPR 2022): the lineage from FCNs to set prediction, every equation derived, proofs of the properties the design relies on, the exact training recipe, and the ablations ranked by what they bought."
pubDate: 2026-07-07
tags: [computer-vision, segmentation, transformers, paper-dissection]
math: true
series: "Peter's Patches"
part: 1
---

> **TL;DR.** Mask2Former [[Cheng et al. 2022](#ref-cheng2022)] made a single architecture beat the best specialized models on panoptic, instance, and semantic segmentation at the same time: 57.8 PQ and 50.1 AP on COCO, 57.7 mIoU on ADE20K. It also trained six times faster than its predecessor in a third of the memory. The mechanism is not scale. It is a rewired Transformer decoder whose cross-attention is masked to each query's own predicted foreground, a coarse-to-fine feeding schedule, three zero-cost optimization changes, and a point-sampled loss. This post is written to be the reference on the paper: the lineage of every idea, every equation derived rather than stated, short proofs of the properties the design quietly relies on, the complete recipe, and the ablations ranked by what they actually bought.

This is the first entry in **Peter's Patches**, a series where I take one influential paper and rebuild it in public until nothing in it is taken on faith. Claims trace to the paper (arXiv:2112.01527) and its appendices. Where a statement is my own reading, or an implementation detail from the official repository rather than the paper, I say so.

**Contents.** [0. Background](#0-the-background-you-need) · [1. The problem, formally](#1-the-segmentation-problem-formally) · [2. Two paradigms](#2-origins-two-paradigms) · [3. Set prediction](#3-the-set-prediction-machinery) · [4. The meta-architecture](#4-the-meta-architecture) · [5. Masked attention](#5-masked-attention) · [6. Multi-scale features](#6-feeding-the-decoder) · [7. Decoder rewiring](#7-rewiring-the-decoder-layer) · [8. Point-sampled losses](#8-losses-match-on-points-train-on-points) · [9. The recipe](#9-the-full-training-recipe) · [10. Results](#10-results-worth-remembering) · [11. Ablations, ranked](#11-what-actually-mattered) · [12. Limitations](#12-limitations-read-honestly) · [13. Lineage forward](#13-where-it-went-next) · [14. Implementation](#14-implementation-corner) · [15. Test yourself](#15-test-yourself) · [16. References](#16-references) · [17. Citation](#17-citation)

## 0. The background you need

Start here if any of this is new. Skip to §1 if words like attention, embedding, and loss already feel comfortable. The notation table below is a reference to return to, not something to memorize.

**Images, pixels, and the job.** An image is a grid of pixels. Segmentation decides which pixels belong together and what the resulting groups are. §1 makes the three meanings of "belong together" precise.

**Feature maps.** A backbone network converts the image into a much smaller grid where each cell holds a vector summarizing a whole patch of the original. That grid is a feature map. Stride 32, written $1/32$, means the grid is 32 times smaller per side, so a 1024 by 1024 image becomes a 32 by 32 summary. Hold onto one consequence: a small, distant object can simply vanish at that scale, because no cell is left to represent it. That single fact drives all of §6.

**Embeddings and the dot product.** The summary vectors are embeddings: vectors whose geometry encodes meaning, arranged so similar things point in similar directions. The dot product $u^\top v$ is the similarity meter. It is large when two vectors agree and near zero when they are unrelated. Keep this close, because the architecture's central trick (§4) is almost embarrassingly simple. A segment's mask is nothing but the dot product between one vector and every pixel's vector, squashed into $[0,1]$.

**Softmax.** Given a list of scores, softmax exponentiates each one and divides by the total. The result is a set of positive weights that sum to 1, a probability distribution over options. Two properties matter here. First, softmax never outputs an exact zero. Every option keeps a sliver of weight, however unpromising. Second, it only cares about score differences, and it renormalizes whatever survives. The first property causes the central pathology of this paper (§5.1). The second is exactly what the fix exploits (§5.2).

**Attention, in one honest paragraph.** Attention is a soft, differentiable dictionary lookup. A query asks a question. Every location in the image offers a key, which says how relevant that location is to the question, and a value, which is what the location contains. Score each location by the dot product of query and key, softmax the scores into weights, and return the weighted average of the values. That is the whole mechanism. When queries read from the image it is called cross-attention. When a set of tokens reads from each other it is called self-attention.

**Transformers and decoders.** A Transformer layer is attention plus a small per-token network, wrapped in a residual connection: keep what you had, add what you just learned. The decoder in this post is a stack of such layers in which 100 learned query tokens repeatedly read from the image and confer among themselves. After nine layers, each query has become a description of one segment.

**Losses, gradients, training.** A loss is a single number measuring how wrong the current output is. Training nudges every weight downhill on that number. The under-appreciated consequence: a network is shaped less by its wiring than by what exactly you choose to measure. Most of this paper's cleverness lives in the measuring (§3, §8).

**IoU.** Intersection over union: the overlap of two regions divided by their combined area. It is 1 when they are identical and 0 when they are disjoint. This is the standard ruler for "are these the same region," and it sits inside every metric in §1.

**Sets versus lists.** A list has an order. A set does not. The model's 100 guesses come out in arbitrary order, and a fair grader must not care about that order. This innocent-sounding fact forces all of the machinery in §3.

**How to read this post.** Every heavy section opens in plain words before any symbol appears. Proofs are short, boxed off, and skippable on a first pass. Anything that is my interpretation rather than the paper's claim is flagged as commentary. Read it twice: once for the story, once for the math.

## Notation

| symbol | meaning |
|---|---|
| $I \in \mathbb{R}^{3\times H\times W}$ | input image; $H_l \times W_l$ is the size of the feature map used at decoder layer $l$ |
| $N$ | number of object queries (100 by default, 200 for the largest models) |
| $K$, $\varnothing$ | number of classes, and the "no object" class appended to them |
| $(m_i, c_i)$ | prediction $i$: a soft mask $m_i \in [0,1]^{H\times W}$ and a class $c_i \in \{1,\dots,K\}\cup\{\varnothing\}$ |
| $\hat p_i(c)$ | predicted probability that query $i$ has class $c$ |
| $\mathbf{X}_l \in \mathbb{R}^{N\times C}$ | query features after decoder layer $l$; $\mathbf{X}_0$ are the learnable input query features |
| $\mathbf{Q}_l, \mathbf{K}_l, \mathbf{V}_l$ | attention projections; $\mathbf{K}_l,\mathbf{V}_l \in \mathbb{R}^{H_lW_l\times C}$ come from image features |
| $M_{l}$ | binarized mask predictions of layer $l$ (threshold $0.5$), resized to the layer's resolution |
| $\mathcal{M}_{l}$ | the additive attention mask built from $M_{l}$: $0$ on foreground, $-\infty$ elsewhere |
| $\mathcal{E}_{\text{pixel}}$ | per-pixel embeddings from the pixel decoder, at stride 4 |
| $\sigma(\cdot)$ | the logistic sigmoid |
| $\lambda_{\text{ce}}, \lambda_{\text{dice}}, \lambda_{\text{cls}}$ | loss weights: $5.0$, $5.0$, $2.0$ (and $0.1$ on $\varnothing$) |
| $K_{\text{pt}}$ | number of sampled points for mask losses: $12{,}544 = 112^2$ |

## 1. The segmentation problem, formally

Image segmentation asks which pixels belong together. The interesting part is that "belong together" admits several meanings. Fix a category set $\mathcal{C} = \mathcal{C}_{\text{th}} \sqcup \mathcal{C}_{\text{st}}$, split into countable *things* (cars, people) and amorphous *stuff* (road, sky), following [[Kirillov et al. 2019a](#ref-kirillov2019pan)]. The three tasks are three output spaces over the same pixels.

**Semantic segmentation** is a map $f: \Omega \to \mathcal{C}$ on the pixel grid $\Omega$. One label per pixel, no identities, so two adjacent cars fuse into one car region. Its metric averages region overlap per class:

$$
\text{mIoU} = \frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}} \frac{|P_c \cap G_c|}{|P_c \cup G_c|},
$$

where $P_c$ and $G_c$ are the predicted and ground-truth pixel sets of class $c$ [[Everingham et al. 2015](#ref-everingham2015)].

**Instance segmentation** outputs a set of scored masks over things only, $\{(m_i, c_i, s_i)\}$, evaluated by mask AP. For each class, predictions are ranked by score. A prediction counts as a true positive when its mask IoU with an unclaimed ground truth exceeds a threshold $\tau$, and AP is the area under the resulting precision-recall curve, averaged over $\tau \in \{0.50, 0.55, \dots, 0.95\}$ in the COCO style [[Lin et al. 2014](#ref-lin2014)]. Note what the metric quietly demands: calibrated ranking, not just good masks. That will matter in post-processing (§9).

**Panoptic segmentation** [[Kirillov et al. 2019a](#ref-kirillov2019pan)] unifies both. Every pixel receives a class and an instance id, with identities on things and plain categories on stuff. Predicted and ground-truth segments are matched, and quality is

$$
\text{PQ} = \frac{\sum_{(p,g)\in \mathit{TP}}\text{IoU}(p,g)}{|\mathit{TP}| + \tfrac12|\mathit{FP}| + \tfrac12|\mathit{FN}|}
=
\underbrace{\frac{\sum_{(p,g)\in \mathit{TP}}\text{IoU}(p,g)}{|\mathit{TP}|}}_{\text{SQ}}
\times
\underbrace{\frac{|\mathit{TP}|}{|\mathit{TP}| + \tfrac12|\mathit{FP}| + \tfrac12|\mathit{FN}|}}_{\text{RQ}},
$$

where a match requires $\text{IoU} > 0.5$. The factorization into segmentation quality and recognition quality is immediate (multiply and divide by $|\mathit{TP}|$). The matching rule hides a small theorem that makes PQ well defined in the first place.

**Lemma (matches are unique).** *If predicted segments are pairwise disjoint, as the panoptic format requires, then each ground-truth segment $g$ has $\text{IoU} > 0.5$ with at most one prediction.*

**Proof.** Suppose $\text{IoU}(p, g) > \tfrac12$. Since $p \cup g \supseteq g$, we get

$$
|p \cap g| > \tfrac12\,|p \cup g| \ge \tfrac12\,|g|.
$$

So any matching prediction claims more than half of $g$. Two disjoint predictions $p_1, p_2$ would claim disjoint subsets of $g$, each larger than $\tfrac12|g|$, so together more than $|g|$ pixels inside $g$. That is impossible. $\blacksquare$

Above the $0.5$ threshold, matching is therefore unambiguous, and greedy matching is exact. Keep this in contrast with training-time matching (§3), where predictions overlap freely, costs are soft, and a genuine assignment problem appears.

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: one scene regrouped under the three segmentation semantics">
<source data-src="/assets/m2f/query_becomes_segment.webm" type="video/webm">
<source data-src="/assets/m2f/query_becomes_segment.mp4" type="video/mp4">
</video>
<figcaption>Fig. 1. Query slots claim segments, a prism separates the what from the where, and the same claimed fields regroup three ways: semantic, instance, panoptic. Nothing re-runs. Only the grouping changes.</figcaption>
</figure>

The observation that motivates the whole research program: these tasks differ only in the semantics of grouping. Yet by 2021 each had its own architecture family, its own tricks, and its own hardware optimizations. Triplicated effort, and specializations that provably do not transfer, as we see next.

## 2. Origins: two paradigms

### 2.1 Per-pixel classification (2015 onward)

The FCN of [[Long et al. 2015](#ref-long2015)] recast semantic segmentation as dense classification. A convolutional network ending in a $1{\times}1$ classifier head outputs $\hat y \in \mathbb{R}^{|\mathcal{C}|\times H\times W}$, trained with per-pixel cross-entropy

$$
\mathcal{L} = -\frac{1}{|\Omega|}\sum_{x\in\Omega}\log \hat p_x(g_x),
$$

where $g_x$ is the true class of pixel $x$. The lineage after it is a search for context: dilated convolutions and pyramid pooling in DeepLab and PSPNet [[Chen et al. 2018](#ref-chen2018), [Zhao et al. 2017](#ref-zhao2017)], self-attention variants [[Wang et al. 2018](#ref-wang2018), [Fu et al. 2019](#ref-fu2019)], and finally pure-Transformer per-pixel models like Segmenter and SegFormer [[Strudel et al. 2021](#ref-strudel2021), [Xie et al. 2021](#ref-xie2021)].

Why this family cannot do instances is worth stating precisely, because it is the formal reason universal architectures exist. A per-pixel classifier's output space is $\mathcal{C}^{\Omega}$, a fixed product of per-pixel labels. Instance segmentation outputs a set of variable size whose elements carry identities that are pure bookkeeping: swapping the names "car 1" and "car 2" gives the same answer. A function into $\mathcal{C}^\Omega$ has no variable for identity at all. Bolting on instance ids as extra channels fails, because any fixed channel-to-identity assignment is arbitrary, and the network would be punished for producing a correct answer in a different order. The disease has a name: the output is invariant under a symmetric group action, so the loss must be too, and per-pixel losses are not. Detection had solved this around 2015 with anchors plus non-maximum suppression, which imposes an artificial order and then de-duplicates. That is exactly the hand-tuned machinery DETR was built to delete.

### 2.2 Mask classification (2017 onward)

The alternative lineage outputs segments directly. **Mask R-CNN** [[He et al. 2017](#ref-he2017)] predicts a binary mask per detected box. That is mask classification, but tethered to boxes, which caps it at things and makes stuff awkward. **Max-DeepLab** [[Wang et al. 2021](#ref-wang2021)] first made mask prediction end-to-end with a Transformer, for panoptic segmentation specifically. The general form arrived with **DETR** [[Carion et al. 2020](#ref-carion2020)]: represent each potential object as a learned query vector, decode all $N$ of them in parallel, and make the loss order-free via bipartite matching (§3). DETR was a detector, but its panoptic extension already predicted masks from queries. **MaskFormer** [[Cheng et al. 2021](#ref-cheng2021)], by the same first author as Mask2Former, then showed the sharp result: a DETR-style mask classifier is not just a way to unify tasks, it beats per-pixel classifiers at semantic segmentation itself. **K-Net** [[Zhang et al. 2021](#ref-zhang2021)] pushed the set-prediction view into instance segmentation with dynamic kernels.

Formally, mask classification predicts

$$
\{(m_i, c_i)\}_{i=1}^{N}, \qquad m_i \in [0,1]^{H\times W},\quad c_i \in \{1,\dots,K\}\cup\{\varnothing\},
$$

and the three tasks fall out by interpretation: segments as categories (semantic), segments as things with identity (instance), or both (panoptic). One architecture, one loss, three annotation schemes.

So by late 2021 universal architectures existed, and an awkward fact sat in the Mask2Former introduction: researchers kept building specialists anyway. Three numbers explain why. Accuracy: the best universal instance result trailed the best specialist by over 9 AP (MaskFormer 40.1 against Swin-HTC++ 49.5 [[Chen et al. 2019](#ref-chen2019), [Liu et al. 2021](#ref-liu2021)]). Compute: MaskFormer needed 300 epochs where HTC++ needed 72 to do better. Memory: full-resolution mask losses meant one image per 32 GB GPU. Mask2Former's thesis: the paradigm was right, the decoder and the recipe were wrong.

## 3. The set-prediction machinery

This section is the mathematical heart shared by DETR, MaskFormer, and Mask2Former. The 2022 paper changes where the losses are evaluated (§8) but inherits this structure, so we do it properly once.

### 3.1 Matching as an assignment problem

**In plain words.** The model always answers with 100 guesses, handed over in no particular order, like a student submitting answers on unnumbered index cards. Before you can grade, you must decide which card was meant for which question, and fairness demands the pairing that flatters the student most. That pairing is the matching. The Hungarian algorithm finds the best one. Everything afterwards is ordinary grading.

Pad the ground truth with $\varnothing$ entries up to size $N$, so predictions and targets are both $N$-element sets. Write $\text{cost}(i, j)$ for the cost of pairing prediction $i$ with target $j$:

$$
\text{cost}(i,j) = -\,\hat p_{i}(c^{\text{gt}}_j) + \mathbb{1}\!\left[c^{\text{gt}}_j \ne \varnothing\right]\Big(\lambda_{\text{ce}}\,\mathcal{L}_{\text{ce}}\big(m_{i}, m^{\text{gt}}_j\big) + \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}}\big(m_{i}, m^{\text{gt}}_j\big)\Big).
$$

This is the DETR convention: raw probability rather than log probability for the class term, plus the same mask terms used in training, active only for real segments. An assignment is a permutation $\sigma \in S_N$, with total cost

$$
\mathcal{C}(\sigma) = \sum_{j=1}^{N} \text{cost}(\sigma(j),\, j),
$$

and training uses the optimal assignment $\hat\sigma = \arg\min_{\sigma} \mathcal{C}(\sigma)$. The Hungarian algorithm computes it exactly [[Kuhn 1955](#ref-kuhn1955), [Munkres 1957](#ref-munkres1957)], in $O(N^3)$ time in the implementations behind `scipy.optimize.linear_sum_assignment`. At $N = 100$ the solve takes microseconds and is never the bottleneck. Building the cost matrix is (§8).

**Proposition (the matched loss ignores prediction order).** *Let $\mathcal{L}(\hat y) = \min_{\sigma\in S_N} \mathcal{C}(\sigma; \hat y)$. Relabeling the predictions by any permutation $\pi$ leaves $\mathcal{L}$ unchanged.*

**Proof.** Relabeling replaces prediction $i$ by prediction $\pi(i)$, so the cost of assignment $\sigma$ under the relabeled predictions is

$$
\sum_{j=1}^{N} \text{cost}\big(\pi(\sigma(j)),\, j\big) = \mathcal{C}(\pi \circ \sigma).
$$

As $\sigma$ ranges over all of $S_N$, so does $\pi \circ \sigma$, because left multiplication by a fixed $\pi$ is a bijection of the group. Minimizing over $\sigma$ on both sides gives the same value. $\blacksquare$

Two lines, but it is the load-bearing property. The storage order of queries carries no information, so the loss must not see it, and with matching it provably does not. It also explains why matching quality sits upstream of everything else. The assignment decides which ground truth each query's gradient comes from, and a wrong pairing trains a query toward the wrong target with full confidence. Hold that thought for §8, where improving only the cost estimates is worth more than 2 AP.

One-to-one matching, as opposed to greedy nearest-target, matters for a subtler reason. Greedy lets two queries claim the same object and leaves another object orphaned. The global assignment forbids duplicate claims by construction, and that is what lets the trained model drop non-maximum suppression entirely.

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: Hungarian matching as cords untangling in segment space">
<source data-src="/assets/m2f/hungarian_matching.webm" type="video/webm">
<source data-src="/assets/m2f/hungarian_matching.mp4" type="video/mp4">
</video>
<figcaption>Fig. 2. Predictions and ground truths as points in segment space, assignments as cords, total cost as a coiled rope that shortens with every swap. The shuffle at the end moves nothing: the loss sees a set.</figcaption>
</figure>

### 3.2 The loss terms, with their gradients

The mask loss is binary cross-entropy plus Dice,

$$
\mathcal{L}_{\text{mask}} = \lambda_{\text{ce}}\,\mathcal{L}_{\text{ce}} + \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}},
\qquad \lambda_{\text{ce}} = \lambda_{\text{dice}} = 5,
$$

and the total loss adds classification, $\mathcal{L} = \mathcal{L}_{\text{mask}} + \lambda_{\text{cls}}\,\mathcal{L}_{\text{cls}}$ with $\lambda_{\text{cls}} = 2$ on matched queries and $0.1$ on $\varnothing$. Why these terms is best seen from their gradients.

**BCE.** At a point $x$ with mask logit $z_x$, prediction $m_x = \sigma(z_x)$, and target $g_x \in \{0,1\}$, the loss is $\mathcal{L}_{\text{ce}} = -\big[g_x \log m_x + (1-g_x)\log(1-m_x)\big]$. Using $\frac{d}{dz}\log\sigma(z) = 1 - \sigma(z)$ and $\frac{d}{dz}\log(1-\sigma(z)) = -\sigma(z)$,

$$
\frac{\partial \mathcal{L}_{\text{ce}}}{\partial z_x} = \sigma(z_x) - g_x.
$$

Clean, well conditioned, and independent per point. That independence is also its weakness: summed over a region, the total gradient scales with the region's area, so large segments shout and small ones whisper.

**Dice** [[Milletari et al. 2016](#ref-milletari2016)], in its soft form (smoothing constants dropped for clarity):

$$
\mathcal{L}_{\text{dice}}(m,g) = 1 - \frac{2\sum_x m_x g_x}{\sum_x m_x + \sum_x g_x}.
$$

Differentiate with the quotient rule, writing $S = \sum_x m_x + \sum_x g_x$ and $O = \sum_x m_x g_x$:

$$
\frac{\partial \mathcal{L}_{\text{dice}}}{\partial m_x} = -\,\frac{2\,g_x\,S - 2\,O}{S^2}.
$$

Read the denominator. Every point's gradient is normalized by the squared region size, so the total gradient a segment receives is roughly independent of its area. The scale invariance is exact in the following sense: tile $k$ disjoint copies of the same prediction and target pattern, and every sum scales by $k$, leaving the loss unchanged. Dice gives a ten-pixel duckling the same voice as the sky. Its price is vanishing gradients when the overlap $O$ is near zero. Summing the two losses is the standard hedge: BCE supplies signal everywhere, Dice supplies fairness across scales.

**What happened to focal loss?** MaskFormer used focal loss [[Lin et al. 2017](#ref-lin2017)] with weight $20$. Mask2Former reverts to plain BCE at weight $5$. The paper states the change without argument. My reading, flagged as commentary: focal loss exists to fight extreme foreground-background imbalance under dense evaluation, and §8's point sampling removes most of that imbalance at the source, letting the simpler and better-conditioned loss win.

**The $\varnothing$ down-weighting.** With $N = 100$ queries and typically 5 to 20 real segments, "no object" outnumbers real matches by roughly 5 to 15 times. At full weight, "predict nothing" becomes the dominant gradient. The down-weighting to $0.1$ is the cheap medicine for the same disease focal loss treats in dense detectors, inherited verbatim from DETR.

**Deep supervision.** The full loss is applied at every one of the 9 decoder layers, and once more on the pre-decoder predictions from $\mathbf{X}_0$. Ten supervised heads per forward pass. In most papers auxiliary losses are an optimization nicety. Here they are structural, because intermediate masks moonlight as attention masks (§5), and "be a useful attention gate" is otherwise never optimized.

## 4. The meta-architecture

Mask2Former keeps MaskFormer's three-part skeleton exactly. A **backbone** (ResNet [[He et al. 2016](#ref-he2016)] or Swin [[Liu et al. 2021](#ref-liu2021)]) extracts low-resolution features. A **pixel decoder** upsamples them into a feature pyramid ending in per-pixel embeddings $\mathcal{E}_{\text{pixel}}$ at stride 4. A **Transformer decoder** processes $N$ query vectors against the image features. Each output query $q_i$ yields a class through a linear head, and a mask through a small MLP followed by a dot product with every pixel embedding:

$$
c_i \sim \operatorname{softmax}(W_{\text{cls}}\, q_i), \qquad
m_i(x) = \sigma\big(\operatorname{MLP}(q_i)^\top\, \mathcal{E}_{\text{pixel}}(x)\big).
$$

A query is therefore a slot that becomes one segment. One vector simultaneously determines what (the class head) and where (its inner product with the embedding field). Masks are decoded at stride 4 and bilinearly upsampled. MaskFormer instantiated this skeleton with an FPN pixel decoder [[Lin et al. 2017b](#ref-lin2017fpn)] and six standard Transformer decoder layers attending over a single stride-32 map. Every one of Mask2Former's contributions is a surgical change inside this fixed skeleton, which is why its ablations decompose so cleanly (§11).

## 5. Masked attention

**In plain words.** Attention lets a query take a weighted average of the entire image, and because the background is vast, the average comes back mostly background. Masked attention forbids the query from averaging anywhere outside the region it currently believes its object occupies, and lets that belief improve layer by layer. The rest of this section is the how, the why, and the proof that it does what it claims.

### 5.1 The pathology, quantified

A standard decoder layer updates queries by cross-attention over the whole feature map:

$$
\mathbf{X}_l = \operatorname{softmax}\!\big(\mathbf{Q}_l \mathbf{K}_l^{\top}\big)\,\mathbf{V}_l + \mathbf{X}_{l-1},
\tag{1}
$$

with $\mathbf{Q}_l = f_Q(\mathbf{X}_{l-1})$ and $\mathbf{K}_l, \mathbf{V}_l$ linear images of the features at resolution $H_l \times W_l$. The paper writes Eq. 1 without the $1/\sqrt{d}$ temperature and the multi-head split for readability. The implementation is standard multi-head attention with both. Worth knowing before you reimplement from the paper alone.

Nothing restricts where a query looks, and two convergence studies of DETR had already indicted exactly this [[Gao et al. 2021](#ref-gao2021), [Sun et al. 2021](#ref-sun2021)]: it takes hundreds of epochs for cross-attention to learn to localize. Mask2Former's appendix adds a blunt measurement of the end state. Even after convergence, averaged over COCO val, only about 20 percent of attention mass lands on the foreground of the segment each query predicts.

The mechanism deserves a short derivation, because it recurs across deep learning. Model the logits as two spikes: $n_f$ foreground locations at $\mu_f$ and $n_b$ background locations at $\mu_b$. The softmax mass on the foreground is

$$
\frac{n_f\, e^{\mu_f}}{n_f\, e^{\mu_f} + n_b\, e^{\mu_b}}
= \frac{1}{1 + \dfrac{n_b}{n_f}\, e^{-(\mu_f - \mu_b)}}.
$$

An object covering 2 percent of the image gives $n_b/n_f = 49$. Even with a healthy logit margin $\mu_f - \mu_b = 2$, the foreground share is $1/(1 + 49\,e^{-2}) \approx 0.13$. Softmax never outputs zero, and the background wins by headcount: thousands of individually negligible weights, integrated over a vast area, dominate the pooled value. This is a heuristic, since real logits are not two spikes, but it lands within a few points of the measured 20 percent.

### 5.2 The mechanism, and why $-\infty$ specifically

The hypothesis is almost provocative: a query does not need global image context at all. Local features from its own segment suffice to update it, and coordination between objects can flow through self-attention between the queries, which are $N = 100$ tokens rather than $H_l W_l \approx 10^4$. Masked attention implements this as an additive mask inside the softmax:

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

where $M_{l-1}$ is the same query's mask from the previous layer, thresholded at $0.5$ after the sigmoid and bilinearly resized to the current attention resolution. $M_0$ comes from the input queries $\mathbf{X}_0$, which is only meaningful because $\mathbf{X}_0$ is learnable and directly supervised (§7).

**Proposition (additive $-\infty$ masking is exact renormalization).** *Let $z \in \mathbb{R}^{n}$ be logits and $S \subseteq \{1,\dots,n\}$ the allowed set, with $\mathcal{M}_i = 0$ for $i\in S$ and $\mathcal{M}_i = -\infty$ otherwise. Then*

$$
\operatorname{softmax}(z + \mathcal{M})_i = \frac{e^{z_i}\,\mathbb{1}[i\in S]}{\sum_{j\in S} e^{z_j}},
$$

*which is exactly the softmax of the original logits restricted to $S$.*

**Proof.** With the convention $e^{-\infty} = 0$, the numerator $e^{z_i + \mathcal{M}_i}$ equals $e^{z_i}$ for $i \in S$ and $0$ otherwise. The denominator is $\sum_j e^{z_j + \mathcal{M}_j} = \sum_{j\in S} e^{z_j}$. Divide. $\blacksquare$

Trivial, but it is the entire design distinction between Mask2Former and its neighbors. Zeroing weights after the softmax breaks normalization: the surviving weights sum to less than 1, so the update shrinks by whatever mass was discarded. Replacing attention with a plain average over the mask, which is K-Net's mask pooling, discards the learned ranking within the region. Masked attention keeps a proper, learned distribution over the foreground. The ablation prices these choices on COCO instance AP: plain cross-attention 37.8, SMCA's soft Gaussian prior 37.9, mask pooling 43.1, masked attention 43.7. Constraint helps enormously, 5.9 AP over none, and the renormalized, learned constraint beats uniform averaging by a further 0.6.

**Origins of the trick.** Additive $-\infty$ masking is not new. It is the mechanism of causal masking in the original Transformer decoder [[Vaswani et al. 2017](#ref-vaswani2017)], where a fixed triangular mask hides the future. Mask2Former's contribution is what generates the mask: not a fixed structural pattern but a predicted, spatial, per-query region, refined online by the network's own output. Read recursively, equations (2) and (3) define a loop in which prediction and attention bootstrap each other nine times. The mask decides where the query reads. What it reads improves the mask. The improved mask sharpens the next read.

<figure class="viz">
<video autoplay loop muted playsinline preload="auto" width="1920" height="1080" aria-label="Animation: masked attention as a stencil cutting background strands while survivors thicken">
<source src="/assets/m2f/masked_attention.webm" type="video/webm">
<source src="/assets/m2f/masked_attention.mp4" type="video/mp4">
</video>
<figcaption>Fig. 3. Attention as light. The open drink returns mostly background. The stencil, which is the previous layer's mask, cuts the outside strands, and the survivors thicken until the beam's total width is what it was. Renormalization, made visible.</figcaption>
</figure>

### 5.3 Stability, gradients, and the guard rail

Three properties keep the recursion from eating itself. First, the residual path in (2) preserves the pre-read state, so one bad read from a wrong region cannot erase a query. Second, masks are re-predicted and re-supervised at every layer, so the attendable region can move. Deep supervision (§3.2) is what trains masks to be good gates, and it has to be, because the gate itself is non-differentiable. Thresholding kills gradients through $\mathcal{M}$, so the learning signal for mask quality arrives only through the per-layer auxiliary losses. Third, an implementation detail from the official repository, absent from the paper, without which training produces NaNs within minutes: if a query's thresholded mask is empty at some scale, every logit is $-\infty$ and the softmax is $0/0$. The code detects such rows and flips them to fully unmasked, quietly falling back to plain cross-attention for that query at that layer:

```python
attn_mask = (mask_logits.sigmoid() < 0.5).flatten(2)   # True = forbidden
attn_mask[attn_mask.all(dim=-1)] = False               # the guard rail
```

### 5.4 Did it work? The paper's own forensics

The appendix analysis is unusually direct. Foreground attention mass rises from about 0.20 to about 0.59, averaged over COCO val and over scales. Not to 1.0, because masks are imperfect and independently thresholded per resolution. Visualized attention for a cat query moves from a diffuse map whose maximum sits outside the cat to a map pinned on it. Layer-wise, a single masked-attention layer already outperforms nine stacked cross-attention layers on panoptic quality. Convergence follows: roughly 25 epochs with standard augmentation, roughly 50 with large-scale jittering, against MaskFormer's 300 and DETR's 500. And in the remove-one-component ablation, masked attention is the paper in one number. Deleting it costs 5.9 AP, 4.8 PQ, and 1.7 mIoU. The damage is largest on instance-level tasks, which fits the mechanism: "attend only to your own segment" is most valuable where the failure mode is confusing identical-class neighbors, and least valuable where identity does not exist.

## 6. Feeding the decoder

### 6.1 The cost structure

Cross-attention cost per layer scales with the token count of the feature map. Building $\mathbf{Q}\mathbf{K}^\top$ and applying it to $\mathbf{V}$ is $O(N \cdot H_lW_l \cdot C)$. At a $1024^2$ input, the pyramid gives $32^2 = 1024$ tokens at stride 32, $64^2 = 4096$ at stride 16, and $128^2 = 16384$ at stride 8, a factor of 16 between coarsest and finest. MaskFormer stayed at stride 32 everywhere. A distant pedestrian at that stride is a single cell, and it shows in MaskFormer's small-object numbers.

### 6.2 The schedule

Mask2Former's answer is a schedule, not a module. Feed one scale per decoder layer, coarse to fine: layer 1 sees $1/32$, layer 2 sees $1/16$, layer 3 sees $1/8$, and the three-layer pattern repeats $L = 3$ times for nine layers. Count token-layer pairs per forward pass. The round-robin touches $3 \times (1024 + 4096 + 16384) = 64{,}512$. Feeding all scales to every layer touches $9 \times 21{,}504 = 193{,}536$, exactly three times more, and the ablation shows it buys nothing: naive multi-scale reaches 44.0 AP at 247 GFLOPs, round-robin 43.7 at 226, single-scale stride 8 everywhere 44.0 at 239. The schedule keeps essentially all of the benefit while touching high-resolution tokens in a third of the layers. Removing high-resolution features altogether costs 2.2 AP, the second-largest single factor after masked attention.

Each scale's features must tell the layer where and which rung. Positions use DETR's 2-D sinusoidal embedding: for each axis and channel pair $i$,

$$
\text{PE}(\text{pos}, 2i) = \sin\!\big(\text{pos}/10000^{2i/(C/2)}\big),\qquad
\text{PE}(\text{pos}, 2i{+}1) = \cos\!\big(\text{pos}/10000^{2i/(C/2)}\big),
$$

computed separately for $x$ and $y$ and concatenated. The rung is a learnable scale-level embedding $e_{\text{lvl}} \in \mathbb{R}^{1\times C}$ per resolution, borrowed from Deformable DETR [[Zhu et al. 2021](#ref-zhu2021)].

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: three resolution panes and an orb breathing coarse to fine">
<source data-src="/assets/m2f/scales_breathe.webm" type="video/webm">
<source data-src="/assets/m2f/scales_breathe.mp4" type="video/mp4">
</video>
<figcaption>Fig. 4. At 1/32 a small object is not representable. At 1/8 everything is, expensively. The decoder breathes coarse to fine, three times, and the fused-slab alternative sags under its own weight.</figcaption>
</figure>

### 6.3 The pixel decoder is a free variable, and that is a finding

Because masked attention only needs some pyramid, any pixel decoder plugs in. The default is a 6-layer multi-scale deformable attention Transformer (MSDeformAttn) over strides $1/8$, $1/16$, $1/32$, plus a lateral-connected upsample to the stride-4 embedding map. Deformable attention [[Zhu et al. 2021](#ref-zhu2021)] replaces dense attention with a learned sparse sample. For a query feature $z_q$ at reference point $\hat p_q$,

$$
\text{MSDeformAttn}\big(z_q, \hat p_q, \{x^l\}\big) = \sum_{m=1}^{M} W_m \Big[ \sum_{l=1}^{L}\sum_{k=1}^{K} A_{mlqk}\; W'_m\, x^l\big(\phi_l(\hat p_q) + \Delta p_{mlqk}\big) \Big],
$$

where $m$ indexes heads, $l$ levels, and $k$ sampling points. The offsets $\Delta p_{mlqk}$ and the attention weights $A_{mlqk}$, normalized by softmax over $(l,k)$, are linear functions of $z_q$. The map $\phi_l$ rescales the reference point to level $l$, and bilinear interpolation reads the fractional positions. Cost per query is $O(MLK\,C)$, a constant number of taps instead of $\sum_l H_lW_l$, which is what makes a Transformer pixel decoder affordable at stride 8.

The cross-decoder ablation quietly restates the paper's thesis in miniature. FPN scores 41.5 AP. Among classic pyramids, BiFPN is best for instance-level tasks and FaPN is best for semantic [[Tan et al. 2020](#ref-tan2020), [Huang et al. 2021](#ref-huang2021)]. Module design re-fragments by task, which is exactly the disease being cured, and only MSDeformAttn wins across all three tasks at once. A universal model doubles as a testbed: a module is not better until it is better everywhere.

## 7. Rewiring the decoder layer

Three changes, zero extra FLOPs, ablated separately, jointly worth about 1.4 AP, 1.1 PQ, and 0.9 mIoU.

**Masked attention comes first.** The vanilla order is self-attention, then cross-attention, then the feed-forward network. Mask2Former swaps to masked attention, then self-attention, then FFN, on a clean argument: at layer 1 the query features are image-independent parameters, so self-attention among them mixes priors that carry no information about this image. Read first, coordinate after. Reverting the order costs 0.5 AP.

**Query features are learnable and directly supervised.** DETR zero-initializes query features and learns only positional embeddings. Mask2Former makes $\mathbf{X}_0$ learnable and supervises the masks decoded from it before the decoder runs, which is also what makes $M_0$ a meaningful first gate. The ablation is pointed: learnable without supervision scores the same as zero-init (42.9 AP), and with supervision 43.7. Supervision is the ingredient, not learnability. Functionally, the supervised $\mathbf{X}_0$ is a region-proposal network [[Ren et al. 2015](#ref-ren2015)] reborn, and the decoder is an iterative proposal refiner.

**Dropout is removed**, for 0.7 AP. The paper reports the effect without theory. My commentary: attention maps here moonlight as localization signals that gate the next layer's reads, and randomly zeroing them injects noise into exactly the pathway the architecture depends on.

## 8. Losses: match on points, train on points

### 8.1 The memory problem and the estimator

**In plain words.** Comparing two masks pixel by pixel is like weighing an entire beach to learn its average grain size: exact, and absurdly heavy. A few hundred well-chosen scoops give the same answer for a fraction of the effort. Surprisingly, choosing the scoops well also improves the grading itself, not just the bill.

Evaluating BCE plus Dice densely for 100 predictions against all ground truths to build the cost matrix, then again for the matched pairs, across ten supervised heads, is what pinned MaskFormer at one image per 32 GB GPU. Following PointRend [[Kirillov et al. 2020](#ref-kirillov2020)], Mask2Former evaluates all mask losses on $K_{\text{pt}} = 12{,}544 = 112^2$ sampled points instead of full masks.

The statistical footing takes three lines. Let $x_1,\dots,x_{K}$ be drawn independently and uniformly from the grid $\Omega$, and let $\ell(x)$ be a per-point loss. The estimator $\hat{\mathcal{L}} = \frac1K\sum_{k}\ell(x_k)$ satisfies

$$
\mathbb{E}\big[\hat{\mathcal{L}}\big] = \mathbb{E}\big[\ell(x_1)\big] = \frac{1}{|\Omega|}\sum_{x\in\Omega}\ell(x) = \mathcal{L}_{\text{dense}},
\qquad
\operatorname{Var}\big[\hat{\mathcal{L}}\big] = \frac{1}{K}\operatorname{Var}\big[\ell(x_1)\big],
$$

by linearity of expectation and independence. The estimate is unbiased, and its noise shrinks like $1/K$. At $K = 12{,}544$ on masks decoded at stride 4, it is essentially exact while touching around 5 percent of the points.

### 8.2 Two sampling rules for two jobs

The genuinely clever part is that the sampling distribution differs by role.

**Matching cost: one shared uniform set.** All prediction and ground-truth pairs in an image are evaluated on the same uniformly sampled points. Sharing is the point. Every entry of the cost matrix is the same functional of the same sample, so entries are comparable and the argmin is not corrupted by independent sampling noise. With per-pair samples, the assignment would sometimes be decided by luck. With a shared sample, the noise shifts all entries coherently.

**Final loss: per-pair importance sampling.** Each matched pair gets its own points via PointRend's procedure, an official-config detail since the paper defers to the citation: oversample $3 K_{\text{pt}}$ candidates uniformly, keep the 75 percent with the most uncertain predictions, meaning mask probability closest to $0.5$, which is near boundaries, and fill the remaining 25 percent uniformly. Note what this is not. There is no importance-weight correction, so the estimator is deliberately biased toward boundaries. It is a boundary-emphasized objective by design, not a cheaper copy of the dense loss. Precision is spent where the masks actually disagree.

<figure class="viz">
<video data-lazy autoplay loop muted playsinline preload="none" width="1920" height="1080" aria-label="Animation: probes on the shoreline of disagreement balancing a beam">
<source data-src="/assets/m2f/shoreline_probes.webm" type="video/webm">
<source data-src="/assets/m2f/shoreline_probes.mp4" type="video/mp4">
</video>
<figcaption>Fig. 5. All the loss lives on a thin shoreline of disagreement. Weighing the whole map dips the beam. A constellation of probes levels it: same weight, a fraction of the cost. Fair where you compare, concentrated where you learn.</figcaption>
</figure>

### 8.3 The result grid, read carefully

| matching on | training loss on | AP (COCO) | PQ (COCO) | mIoU (ADE20K) | memory |
|---|---|---|---|---|---|
| masks | masks | 41.0 | 50.3 | 45.9 | 18 GB |
| masks | points | 41.0 | 50.8 | 45.9 | **6 GB** |
| points | masks | 43.1 | 51.4 | 47.3 | 18 GB |
| **points** | **points** | **43.7** | **51.9** | **47.2** | **6 GB** |

Point-sampling the training loss is the memory story: a threefold reduction at zero accuracy cost, which is the unbiasedness argument verified. Point-sampling the matching cost is an accuracy story: over 2 AP, on top of everything else. The paper offers no mechanism. My reading, flagged as commentary: dense low-resolution cost matrices are dominated by easy interior and background agreement, near-saturated terms that blur the distinctions between candidate assignments, while a shared sparse sample yields a sharper cost surface. And by the proposition of §3.1, assignment quality sits upstream of every gradient. Either way, the memorable form: the cheap version is also the better version.

## 9. The full training recipe

Papers live or die on recipes, and Mask2Former's appendix is explicit enough to reproduce, including a side-by-side against MaskFormer that §11 will need.

| | MaskFormer | Mask2Former |
|---|---|---|
| optimizer | AdamW [[Loshchilov & Hutter 2019](#ref-loshchilov2019)], lr $10^{-4}$ | AdamW, lr $10^{-4}$ |
| weight decay | $10^{-4}$ | **0.05** |
| backbone lr multiplier | 0.1 | 0.1 |
| schedule (COCO) | 300 epochs | **50 epochs**, lr ×0.1 at 90% and 95% of steps |
| batch | 16 | 16 |
| augmentation | standard scale and crop | **LSJ** [[Ghiasi et al. 2021](#ref-ghiasi2021)]: scale 0.1 to 2.0, fixed $1024^2$ crop |
| mask loss | focal ($\lambda{=}20$) + dice ($\lambda{=}1$), dense | **BCE ($\lambda{=}5$) + dice ($\lambda{=}5$)** on 12,544 points |
| $\lambda_{\text{cls}}$ | 1.0 | 2.0, with 0.1 on $\varnothing$ |
| decoder | 6 layers, SA→CA→FFN, dropout 0.1, stride 32 only, zero-init queries | **9 layers, MA→SA→FFN, no dropout, strides {32,16,8}×3, learnable supervised queries** |

Inference on COCO follows the Mask R-CNN protocol: shorter side 800, longer side at most 1333. Queries: 100 everywhere except 200 for the largest panoptic and instance models, trained 100 epochs. The query ablation shows 100 is best for instance and semantic, 200 helps only panoptic (52.2 against 51.9 PQ, since panoptic scenes hold more segments), and 1000 actively hurts (40.3 AP). Per-dataset settings: Cityscapes [[Cordts et al. 2016](#ref-cordts2016)] uses 90k iterations at 512×1024 crops, ADE20K [[Zhou et al. 2017](#ref-zhou2017)] uses $640^2$ crops, Mapillary Vistas [[Neuhold et al. 2017](#ref-neuhold2017)] uses 300k iterations at $1024^2$ crops.

Post-processing is inherited from MaskFormer. Semantic output is the per-pixel argmax of class-probability-weighted masks, $\arg\max_c \sum_i \hat p_i(c)\, m_i(x)$. Panoptic works the same way, with low-confidence filtering to resolve overlaps into a partition. Instance segmentation needs the calibrated ranking AP demands (§1), so the score is

$$
s_i = \hat p_i(c_i)\cdot \frac{1}{|\{x : m_i(x) > 0.5\}|}\sum_{x:\, m_i(x)>0.5} m_i(x),
$$

class confidence times average mask confidence over the foreground, because a query can be sure about what something is while sloppy about where.

## 10. Results worth remembering

One architecture, per-task training, state of the art everywhere, for the first time:

| task / dataset | Mask2Former (Swin-L) | previous best | margin |
|---|---|---|---|
| Panoptic, COCO val | **57.8 PQ** | MaskFormer 52.7, K-Net 54.6 | +5.1 / +3.2 |
| Instance, COCO val | **50.1 AP** (36.2 boundary AP) | Swin-HTC++ 49.5 (34.1) | +0.6 (+2.1) |
| Semantic, ADE20K val | **57.7 mIoU** (Swin-L, FaPN, multi-scale) | BEiT 57.0 | +0.7 at roughly half the parameters |

At ResNet-50 scale the story is learning efficiency: 51.9 PQ in 50 epochs against MaskFormer's 46.5 in 300, six times faster to a better place, and 43.7 instance AP in 50 epochs against 42.5 for a heavily tuned 400-epoch Mask R-CNN.

Three second-order results carry more information than the headlines. On test-dev instance segmentation, large-object AP reaches 71.2, beating the challenge winner's 67.7 despite the winner's extra data and ensembling, while small-object AP is 29.1 against their 36.6. The paradigm is spectacular on large objects and clearly behind on small ones, the paper's own declared open problem. Boundary AP rises by 2.1 over HTC++ against 0.6 overall, so the stride-4 embedding map pays exactly at mask edges. And the compute-performance frontier genuinely moves, since the lightest Mask2Former beats the heaviest MaskFormer at a quarter of the FLOPs, though honesty requires the throughput footnote: the R50 panoptic model runs 8.6 fps to MaskFormer's 17.6. Multi-scale attention is not free. It is very well spent.

Generalization holds without architectural change: Cityscapes 66.6 PQ, 43.7 AP, 83.3 mIoU with Swin-L, ADE20K panoptic 46.2 PQ, Mapillary Vistas 45.5 PQ and 63.2 mIoU. Competitive with street-scene specialists on their home turf.

## 11. What actually mattered

Every row is a controlled experiment on R50, across all three tasks:

| change (removed or varied) | ΔAP | ΔPQ | ΔmIoU | takeaway |
|---|---|---|---|---|
| remove masked attention | **−5.9** | **−4.8** | −1.7 | the paper, in one number |
| remove multi-scale high-res features | −2.2 | −1.7 | −1.1 | resolution is second, and the schedule makes it affordable |
| point→mask matching cost | −2.7 | −0.5 | −2.4* | the sleeper: assignment quality is upstream of everything |
| remove query supervision | −0.8 | −0.7 | −1.8 | supervision, not learnability, is the ingredient |
| restore dropout | −0.7 | −0.6 | 0.0 | attention maps are localization signals, do not corrupt them |
| vanilla layer order | −0.5 | −0.3 | −0.9 | read the image before talking among queries |

*at fixed point-sampled training loss. The §8.3 grid has the full two-by-two.

The appendix also runs the decomposition every reviewer secretly wants: recipe against architecture. Retraining MaskFormer with Mask2Former's training parameters recovers several AP by itself, so LSJ, the reweighted BCE plus Dice, and the point losses transfer to other models. Swapping in the new decoder while holding the backbone, the FPN pixel decoder, and the recipe fixed adds several more, and the MSDeformAttn default closes the rest of the gap to 43.7. Roughly a third recipe, a third decoder, a third pixel decoder, stated where most papers would let the headline idea absorb all the credit. Steal that habit.

## 12. Limitations, read honestly

**Universal means one architecture, not one checkpoint.** A panoptic-trained Mask2Former evaluated as an instance or semantic model trails per-task training, but the gaps are small and not one-sided. On COCO the panoptic checkpoint gives up about 2 AP on instance while matching semantic. Panoptic training nearly subsumes instance already, and the residual gaps sit in stuff classes. The paper names the next goal, train once for everything, which is exactly what OneFormer later delivered.

**Small objects remain the weak flank**, and the authors concede the pyramid is under-exploited: the round-robin schedule is an efficiency compromise, not a solution. Add two soft costs from the tables: the throughput gap above, and the query-count coupling, where one hyperparameter still quietly encodes how many segments your task tends to produce.

## 13. Where it went next

Mask2Former's decoder became infrastructure. The same group extended it unchanged to video, where masks become spatio-temporal tubes [[Cheng et al. 2021b](#ref-cheng2021vis)]. **OneFormer** [[Jain et al. 2023](#ref-jain2023)] closed the paper's own declared gap, one jointly trained model for all three tasks, by conditioning the same skeleton on a task token. **Mask DINO** [[Li et al. 2023](#ref-li2023)] unified it with DETR-style detection, letting box and mask queries help each other. The query-as-segment abstraction became the substrate for open-vocabulary segmentation, where the fixed classifier is replaced by text embeddings, and the "predict masks, classify separately" philosophy echoes in SAM's promptable, class-agnostic design [[Kirillov et al. 2023](#ref-kirillov2023)], even though SAM's goal differs. Mask2Former itself shipped in Detectron2 [[Wu et al. 2019](#ref-wu2019)] and in `transformers` as `facebook/mask2former-*`, and it remains the baseline every new segmentation paper must beat. Leaderboards have moved since 2022. The skeleton mostly has not.

## 14. Implementation corner

The reference implementation is [`facebookresearch/Mask2Former`](https://github.com/facebookresearch/Mask2Former), and its configs reproduce every table above. The heart, one decoder layer with the two details people miss, in PyTorch-flavored pseudocode:

```python
def decoder_layer(x, feats_l, pos_l, lvl_emb, query_pos, prev_mask_logits, layer):
    # x: (N,B,C) queries · feats_l: (H_l*W_l,B,C) image features at this scale
    m = F.interpolate(prev_mask_logits, size=spatial(feats_l), mode="bilinear")
    attn_mask = (m.sigmoid() < 0.5).flatten(2)      # True = forbidden
    attn_mask[attn_mask.all(dim=-1)] = False        # NaN guard (official repo)
    # bool mask => no gradient through the gate; masks learn via aux losses

    x = norm1(x + cross_attn(q=with_pos(x, query_pos),          # masked attention FIRST
                             k=with_pos(feats_l, pos_l + lvl_emb),
                             v=feats_l, attn_mask=attn_mask))   # no dropout anywhere
    x = norm2(x + self_attn(with_pos(x, query_pos)))            # queries coordinate
    x = norm3(x + ffn(x))

    cls_logits  = class_head(x)                                  # (B,N,K+1)
    mask_logits = einsum("bnc,bchw->bnhw", mask_head(x), pixel_embeddings)  # stride 4
    return x, cls_logits, mask_logits    # supervised here AND gates the next layer
```

Assembly: run the heads once on the learnable $\mathbf{X}_0$ before the loop, giving both an auxiliary target and $M_0$, cycle the three scales for nine layers, and feed all ten class-mask pairs to the Hungarian-matched, point-sampled loss of §8. To feel the paper, reproduce one row of the ablation: set `attn_mask=None` and watch convergence collapse.

## 15. Test yourself

**Prove that adding $-\infty$ before the softmax renormalizes over the allowed set, and say precisely what post-softmax zeroing breaks.** The §5.2 proposition: $e^{z+\mathcal{M}}$ vanishes off $S$, so the weights are the restricted softmax and still sum to 1 with the learned ranking intact. Zeroing after the softmax leaves total weight below 1, so the update shrinks by the discarded mass. Mask pooling replaces the ranking with a uniform average, and the ablation prices that ranking at 0.6 AP.

**Derive the foreground attention share for an object covering fraction $\rho$ of the image with logit margin $\Delta$, and evaluate it at $\rho = 0.02$, $\Delta = 2$.** Share $= 1/\big(1 + \frac{1-\rho}{\rho}e^{-\Delta}\big) = 1/(1+49e^{-2}) \approx 0.13$. That is the headcount pathology of §5.1, and it lands near the measured 20 percent.

**Show the matched loss is permutation-invariant.** §3.1: relabeling predictions turns the cost of $\sigma$ into the cost of $\pi\circ\sigma$, and left multiplication is a bijection of $S_N$, so the minimum is unchanged. Corollary: the storage order of queries is provably information-free.

**Gradients do not flow through a thresholded attention mask. How do masks learn to be good gates?** They do not learn through the gate. Per-layer deep supervision trains every intermediate mask directly. The auxiliary losses are load-bearing.

**Why is the matching sample shared and uniform, while the loss sample is per-pair and boundary-biased?** Shared and uniform makes cost-matrix entries comparable, since every entry is the same functional of the same points, and unbiased, since assignment must be fair. The loss needs no fairness across pairs, so the bias is spent deliberately: uncertainty sampling concentrates gradient where the masks disagree.

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
27. <a name="ref-chen2018"></a>Chen, Papandreou, Kokkinos, Murphy, Yuille. "DeepLab: Semantic Image Segmentation with Atrous Convolution and CRFs." TPAMI 2018.
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
</script>
