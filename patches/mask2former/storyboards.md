# Peter's Patches #1 — Visual rebuild
## 3b1b-grade storyboards + Claude Code prompts for the Mask2Former figures

---

## 0. Recap: where this stands and what changes

**What exists and holds up.** The article (`mask2former-explained.md`, ~6,200 words) is done and is the strongest part of the package: full math, the training recipe, ranked ablations, the NaN guard from the official repo, the interview Q&A. It doesn't change.

**What failed, and why.** Two generations of figures so far. The first were animated diagrams — boxes and arrows fading in, which read as slides. The second animated real computation but still explained through *instruments*: bar charts, percentage meters, running counters. That is the core miss. Grant Sanderson's videos contain almost no charts. His rule, stated plainly: **the geometry is the argument.** You don't show a meter reading 20% → 59%; you show thin strands of light from everywhere braiding into a muddy beam, then a stencil cutting the background strands while the survivors *visibly thicken to conserve the total*. The viewer concludes the renormalization themselves. A chart reports a result; a transformation *is* the reasoning.

**What this document is.** No code. Five storyboards written at the level a director hands an animator — thesis, cast, shot-by-shot beats with timings, the freeze-frame each scene must earn — plus a design constitution derived from the actual character of peteramassih.com (minimal, typographic, one accent, generous space), and copy-paste Claude Code prompts that implement, self-review against freeze-frames, and ship. The flagship is the one you called out: **Hungarian matching** — genuinely the hardest idea in the paper to see, and the one no existing explainer visualizes at all.

**Replacement map.** These five scenes supersede `masked_attention`, `point_sampling`, `task_zoo`, `round_robin`, and `meta_architecture`. The `layerwise_pq` chart may survive as a small *static* figure in §4.3 (a chart is honest when the content is literally a measurement) or be dropped.

---

## 1. The grammar — what makes a figure read as 3b1b

These are hard rules, not vibes. Every prompt below enforces them; every scene is reviewed against them frame by frame.

**1. Geometry is the argument.** If a caption or a chart is doing the explaining, the shot has failed. Ask of any paused frame: could a viewer reconstruct the idea from the shapes alone?

**2. Nothing pops; everything transforms.** Objects enter by being drawn or growing; they change by morphing. The viewer must be able to track any object's identity through the entire scene — identity-through-transformation is how understanding transfers.

**3. Colors are nouns.** One fixed semantic palette for the whole series (§2). A color never decorates; gold always *means* prediction, everywhere, forever. Recoloring is a plot event.

**4. No instruments.** No bar charts, axes, legends, percentages, counters, or gauges. When a magnitude matters, it becomes length, area, thickness, brightness, tautness, or weight. (Numbers may appear exactly once per scene, at the very end, as a small factual residue — never as the mechanism.)

**5. The camera is attention.** Zoom to what is being reasoned about; dim everything else to ~20%; pull wide for consequences. At most one camera move per beat.

**6. Tension, hold, resolve.** Every scene poses its question *visually* — something is visibly strained, tangled, or wrong — holds it for a full beat, then resolves it. The resolution should feel earned, with a soft settle, never a cut.

**7. Conservation makes invisible math visible.** Renormalization = total thickness conserved and redistributed. Unbiasedness = a balance beam leveling. One-to-one assignment = cords that cannot share an endpoint.

**8. Equations are residue, not narration.** If an equation appears, it is assembled at the end *from the scene's own objects* (the stencil flies into the parentheses as "+M"), small, once.

**9. Motion quality.** 60 fps, ease-in-out on everything, beats of 0.8–1.6 s, scene length 30–50 s, final frame held 2 s, loop-friendly where natural. Nothing moves linearly except a deliberate scan.

**10. It must sit natively on the page.** Background, ink, and accent are the site's own tokens, extracted from its CSS — not invented. Captions are set in the site's body face, ≤ 6 words, at most one on screen. Branding: a single small "peteramassih.com · Peter's Patches" bottom-right at 40 % opacity.

---

## 2. The cast — one semantic palette for the whole series

Claude Code extracts the site's real tokens first (background, ink, muted, accent) and binds this cast to hues that live comfortably on that background. The *meanings* are fixed; the exact hex values come from the site.

| role (a noun, never a decoration) | working name |
|---|---|
| predictions / queries — everything the model asserts | **gold** |
| ground truth — everything the data asserts | **green** |
| image features, pixels, keys — the raw world | **slate-blue** |
| cost, strain, error, disagreement | **ember** (used sparingly; ember on screen = something is wrong) |
| the void class ∅ — "no object" | **hollow** (outline-only, no fill) |
| neutral structure, captions | site ink / muted |

Texture rules matching the site's restraint: thin strokes (2–3 px at 1080p), soft-edged fills, no drop shadows, no glow beyond a 1-px halo on gold, whitespace as generous as the homepage's.

---

## 3. Storyboard A — `hungarian_matching` (flagship, ~48 s)

**Thesis in one sentence:** the model outputs a *set*, so the loss must pair predictions with ground truths by **content, not order** — and once paired, every gradient travels along that pairing.

**Cast.** A quiet image strip (three flat silhouettes: two ducks, one dog — same object twice is deliberate, it's what makes matching hard). Six gold soft-edged prediction masks. Three green ground-truth silhouettes. **Cords**: curved connectors whose length stands for cost and whose tautness/ember-tint stands for strain. A faint dotted disc — "segment space" — where masks live as points. A dim shelf at the edge with hollow ∅ rings.

**Shot 1 (0:00–0:07) — the naive loss breaks.** Predictions line up top (their order is just storage order), ground truths bottom. Straight cords connect index-to-index: pred 1 ↔ GT 1, pred 2 ↔ GT 2… Now the top row *shuffles* — gold masks slide past each other, because a set has no order. The cords whip to new partners, stretch, tint ember, and the whole rig visibly strains. Hold the strain a full second. Caption, the only one in the shot: **"same set — different loss?"** The question the scene must answer is now on screen as tension, not text.

**Shot 2 (0:07–0:16) — masks become points.** One gold mask lifts off the image, shrinks, and *condenses into a point*; its silhouette lingers as a ghost for half a beat so the identity transfer is unmissable. All masks and all GTs condense likewise and settle onto the dotted disc — and here is the earned geometry: **lookalikes land near each other.** The duck-shaped prediction drifts in and settles beside the duck GT point; the garbage prediction lands far from everything. Distance now *means* dissimilarity, because the viewer watched similar shapes travel to nearby places. No caption needed.

**Shot 3 (0:16–0:22) — cost is a braid.** Camera pushes in on one cord. It resolves into a **braid of three thin strands** — one gold-ish (does the class match?), two slate (do the shapes overlap? do the regions overlap?). The braid is the composite cost — class + BCE + dice — shown, not itemized. Pull back; every cord is quietly a braid from now on.

**Shot 4 (0:22–0:30) — untangling.** Draw a deliberately bad assignment: cords crossing, long, humming with strain; at the side, all their lengths pour into a single coiled arc — the **total cost as one physical length of rope**. Then the Hungarian step: cord endpoints *slide* (never teleport) along the disc, exchanging partners pairwise, each swap visibly shortening the coil, until the rig settles slack and quiet with a soft snap. One-to-one is enforced by the geometry itself: a point simply cannot hold two cord-ends.

**Shot 5 (0:30–0:36) — the leftovers.** Three gold points end up cordless. Hollow ∅ rings fasten around them and they drift to the dim shelf, connected to it by the *thinnest possible* threads — a faint tug, not a pull. (That whisper-thin thread **is** λ∅ = 0.1: the "no object" signal exists but must not dominate. Never say this on screen; the thinness says it.)

**Shot 6 (0:36–0:42) — the payoff, and the freeze-frame this scene exists for.** The original prediction list reappears at the top and shuffles *violently*. Below, in segment space: **nothing moves.** Cords hold. Coil length identical. Hold two full seconds. Caption: **"the loss sees a set."** This single held frame is permutation invariance, proven visually.

**Shot 7 (0:42–0:48) — gradients ride the cords.** The cords soften into springs. Three slow pulses — three epochs — and each gold point is drawn along its cord toward its green partner; two points re-inflate into thumbnails mid-travel so you watch the masks themselves improving. The ∅ points fade a step further. Final frame: pairs nearly coincident, coil tiny, rig at rest.

**Anti-patterns to reject on sight:** a cost *matrix* of numbers (the standard bad viz), labeled arrows, any mention of O(n³), more than two captions total.

---

## 4. Storyboard B — `masked_attention` (redo, ~42 s, zero numbers)

**Thesis:** cross-attention forces a query to drink the whole image, and the vast dim background out-pours the small bright object; a mask is a stencil that lets it drink only from its own segment — and the stencil and the understanding sharpen each other, layer after layer.

**Cast.** The image as a soft field (a cat-shaped warm region on a cool slate expanse). One gold query orb, hovering. **Light as attention**: the orb illuminates; the field reflects; thin colored strands return to the orb and braid into the beam it drinks. A **stencil** — a translucent plate with a hole — that can slide between orb and image. The orb's own fill color is the plot: it shows what the query currently "is."

**Shot 1 (0:00–0:06) — the drink.** The orb glows; illumination spreads over the *entire* field (it has no choice). Hundreds of hair-thin strands rise back to it — a few warm cat-colored ones, a multitude of cool background ones — and braid into a single returning beam. The orb's fill turns **mud**. Caption: **"it reads everything."**

**Shot 2 (0:06–0:12) — why the background wins.** Camera glides low across the field: the cat is bright but *small*; the background is faint but *endless*. Let the strand bundle make the argument — the cool strands are individually gossamer yet together form almost the whole braid's width. Softmax-never-zero, area-times-brightness, shown as thread-count. No meter. No percentage.

**Shot 3 (0:12–0:20) — the stencil.** The previous layer's mask arrives as a physical stencil — hole roughly cat-shaped, imperfect, a little offset — and slides between orb and field. Strands from outside the hole are **cut**: they retract and die. And the conservation beat, the most important three seconds of the scene: the surviving strands **thicken until the beam's total width is exactly what it was**. Renormalization, shown as conserved thickness redistributing. The engraving on the stencil's opaque region reads only "−∞". The orb's fill clears from mud to warm cat.

**Shot 4 (0:20–0:26) — the loop.** With a cleaner fill, the orb redraws its stencil — the hole *tightens* from blob toward true cat silhouette. Light refocuses. Fill warms further. Three of these breaths, each one smooth inhale-exhale, then an ellipsis (…) and a small "×9" as the breaths continue offscreen. Stencil and understanding co-evolve; that's the paper.

**Shot 5 (0:26–0:32) — the failsafe.** One breath goes wrong: the hole shrinks to nothing. For that layer the stencil simply **dissolves** — the orb drinks openly once, recovers, redraws. (This is the official repo's empty-mask NaN guard, staged in two seconds and never named.)

**Shot 6 (0:32–0:42) — residue.** The scene's objects assemble the equation: the parenthesis draws itself around the strands' origin, the stencil flies in and shrinks to a small **+M**, and `softmax( QKᵀ + M ) V` sits quietly at the bottom, small. Optional single number, final second, muted: *"foreground share, measured: 0.20 → 0.59 (paper, App. C)"* — a residue, not a mechanism. Hold.

**Anti-patterns:** any bar, any percentage during the mechanism, weight heat-grids with legends, more than three captions.

---

## 5. Storyboard C — `query_becomes_segment` (~35 s)

**Thesis:** a query is a slot that *becomes* one segment — its content simultaneously says **where** (a field over the pixels) and **what** (a class) — and the three segmentation tasks are the same machinery under different groupings.

**Cast.** The image field. A file of ~ten identical gold orbs (standing for N = 100; a trailing "…" carries the rest). Each orb casts a faint **affinity field** over the image — literally its dot product with the pixel embeddings, rendered as a soft luminous region. A small prism for the class head.

**Shot 1 (0:00–0:06) — born with opinions.** The orbs enter *before touching the decoder* — and already cast rough, blobby fields over the image. (Learnable, supervised X₀: the queries are proposals from birth. Shown, not said.)

**Shot 2 (0:06–0:16) — crystallization.** The orbs pass through the decoder — drawn as nothing more than a sequence of nine faint gates — and with each gate their fields sharpen and *claim*: this orb's glow collapses onto the left duck, that one onto the dog, each field snapping to a crisp silhouette by gate nine. Two orbs briefly fight over the same duck; self-attention is staged as a single spark between them, after which one yields and drifts toward an unclaimed region. Coordination, in one beat.

**Shot 3 (0:16–0:22) — the prism.** A crystallized orb passes through the prism and refracts into a labeled tint — *duck*. Where-and-what, from one vector: the field was the mask head, the refraction is the class head.

**Shot 4 (0:22–0:35) — one machine, three tasks.** Freeze the claimed scene. Now regroup the *same* fields three ways in place: tint by category only (two ducks merge into one hue — semantic); keep identities, dim the background stuff (instance); keep both (panoptic). Nothing re-runs; only the grouping changes. Caption: **"same machine — three semantics."** Final hold.

**Anti-patterns:** an architecture block diagram, arrows labeled "backbone/pixel decoder" (that diagram already exists in the article as a static figure; this scene is about *meaning*, not plumbing).

---

## 6. Storyboard D — `scales_breathe` (~30 s)

**Thesis:** at 1/32 resolution a small object *does not exist*; at 1/8 everything exists but there is far too much of it — so the decoder breathes: one scale per layer, coarse to fine, three times.

**Cast.** The image as three **glass panes** stacked in depth — 1/32 (frosted almost opaque), 1/16 (hazy), 1/8 (near-clear) — each edge-tinted with its own faint level hue (the learnable level embedding, worn, not labeled). The gold orb from Storyboard C. A tiny duckling in the scene.

**Shot 1 (0:00–0:07) — the vanishing.** Camera dives from the sharp pane toward the frosted one, and the duckling *dissolves on the way down* — at 1/32 it is literally not representable. Hold on the frosted pane: big shapes only. The problem, in one camera move.

**Shot 2 (0:07–0:12) — the temptation.** Snap to the 1/8 pane: the duckling is back — but so is an ocean of texture; render the pane's sheer *density* (a fine lattice shimmering across its whole surface) so staying here feels heavy. Both extremes are now felt.

**Shot 3 (0:12–0:22) — the breath.** The orb takes the schedule: it drinks from the frosted pane (broad shapes settle its field), glides down to the hazy pane (the duckling appears as a smudge in its field), then the clear pane (edges lock). Then again. Then a third time — nine gates total ticking faintly at the edge. Each cycle is one smooth breath; the orb's field visibly gains structure coarse-to-fine, like an image loading progressively.

**Shot 4 (0:22–0:30) — the alternative, rejected.** The three panes try to fuse into one thick slab in front of the orb — and the slab visibly *sags with weight*, dragging the orb's glide to a crawl. The panes separate again; the breath resumes, light. All scales at once buys nothing and costs everything — argued by weight, not FLOPs. Final hold on the breathing loop; this scene should loop seamlessly.

**Anti-patterns:** stride numbers as the focus, arrows from boxes to boxes, any token-count annotation.

---

## 7. Storyboard E — `shoreline_probes` (~32 s)

**Thesis:** the disagreement between a predicted mask and the truth lives on a thin shoreline; you don't need to weigh the whole map to know the shoreline's weight — a handful of well-placed probes measures the same thing — and *where* you may place them differs between matching (fair) and training (efficient).

**Cast.** Two translucent sheets overlaid on a dark table: the green truth silhouette and the gold soft prediction, slightly offset. Where they disagree, the overlap **glows ember** — a thin, irregular shoreline. A balance beam at the side, two pans. Probe **needles**.

**Shot 1 (0:00–0:06) — the shoreline.** The sheets slide over each other and settle; agreement stays dark, disagreement ignites into the shoreline. Camera skims along it. Caption: **"all the loss lives here."**

**Shot 2 (0:06–0:12) — weighing everything.** The entire glowing sheet lifts and settles onto the left pan; the beam dips under it. This is the dense loss: exact, and heavy — the whole map on the scale. (The 18 GB never appears; the *heft* does.)

**Shot 3 (0:12–0:20) — the probes.** A constellation of needles rains uniformly onto a duplicate sheet; each needle that lands on shoreline takes on a spark, the rest stay cold. The sparks alone are swept onto the right pan — and the beam **levels**. Equal weight from a few hundred points: an unbiased estimate, proven by balance, no numbers.

**Shot 4 (0:20–0:26) — two rules for two jobs.** Split screen, same constellation logic twice. Left — *matching*: the **identical** uniform constellation is stamped, like a die, onto three different prediction/truth pairs in a row; same probe pattern everywhere, so their weights can be *compared fairly*. Right — *training*: needles rain, then **magnetize**, sliding across the sheet to crowd the shoreline; wasted cold needles go dim. Fair where you compare; concentrated where you learn.

**Shot 5 (0:26–0:32) — residue.** The needles arrange themselves into a faint 112 × 112 lattice for half a beat — the only number-shaped thing in the scene — then relax. Final frame: leveled beam, glowing shoreline, resting needles. Optional muted residue line: *"12,544 points · same loss · a third of the memory."*

**Anti-patterns:** memory bars, counters converging to a decimal, dot-scatter without the balance beam (the beam *is* the proof).

---

## 8. Where each scene lands in the article

`hungarian_matching` → §8.1 (becomes the section's opening figure; the current §8 figure moves down). `masked_attention` → §4.2 hero (replaces current). `query_becomes_segment` → §2/§3 boundary (replaces `task_zoo` and demotes `meta_architecture` to a static labeled diagram, which is what an architecture wants to be). `scales_breathe` → §5 (replaces `round_robin`). `shoreline_probes` → §8.2 (replaces `point_sampling`). `layerwise_pq` → keep as a small static chart in §4.3 or cut. Update the appendix asset manifest accordingly; captions in the article can stay, lightly reworded to match the new imagery (cords, stencil, breaths, shoreline).

---

## 9. Claude Code prompts

Paste in order, from the site repo root, with this file saved at `patches/mask2former/storyboards.md`.

### Prompt 0 — the constitution (run first, once)

```
You are implementing five Manim CE scenes for peteramassih.com from the
storyboards in patches/mask2former/storyboards.md. Read that file fully first —
sections 1 (grammar), 2 (cast), and the five storyboards. It is the contract.

Setup:
1. Extract the site's real design tokens: find the CSS/theme source
   (globals.css / tailwind config / whatever this repo uses) and read the
   computed background, text ink, muted text, and accent colors. Write
   patches/mask2former/viz/tokens.py exporting BACKGROUND, INK, MUTED, ACCENT,
   plus the semantic cast (GOLD, GREEN, SLATE, EMBER, HOLLOW) chosen to sit
   comfortably on that exact background — if the site is light, the scenes are
   light. Never inline a hex in a scene; only tokens.
2. Scenes render at the site background so exported frames are
   indistinguishable from the page. Typography: the site's body face if
   licensable/loadable in Pango, else the closest system face; captions <= 6
   words, at most one visible at a time.

Hard rules (from storyboards.md §1 — enforce mechanically):
- no bar charts, axes, legends, percentages, counters, or gauges anywhere;
- nothing appears by popping: draw-on, grow, or morph only, and any object's
  identity must be trackable across the scene;
- semantic colors only, from tokens.py;
- one camera move per beat maximum; ease-in-out everywhere; 60 fps; hold the
  final frame 2 s;
- an equation may appear once, at the end, assembled from scene objects;
- a single number may appear only in a scene's final "residue" beat where the
  storyboard explicitly allows it.

Iteration protocol for every scene (never skip):
a. implement from the storyboard, one shot at a time;
b. render draft: manim -ql;
c. export a frame at each storyboard timestamp plus the scene's freeze-frame,
   and REVIEW THE IMAGES: check each shot achieves its stated purpose, nothing
   overlaps, nothing clips, no forbidden instrument crept in;
d. fix and repeat until every freeze-frame passes;
e. final render: manim -qh at 60 fps; export .mp4, .webm
   (libvpx-vp9, crf 32, no audio, target < 2 MB), and a fallback .gif
   (fps 10, width 720, palette-optimized) into static assets at /assets/m2f/.

Deliver scenes in this order, one commit each, pausing after each for my
review: hungarian_matching, masked_attention, query_becomes_segment,
scales_breathe, shoreline_probes. Start now with Prompt 1.
```

### Prompt 1 — `hungarian_matching`

```
Implement Storyboard A (hungarian_matching, ~48 s) from
patches/mask2former/storyboards.md exactly, shot by shot.

Freeze-frames that must pass before you show me anything:
- ~0:05  index-to-index cords under shuffle: visibly strained, ember-tinted,
  and obviously arbitrary;
- ~0:12  a mask mid-condensation into a point, ghost silhouette still legible —
  the identity handoff must be unmissable;
- ~0:28  mid-untangling: two cord endpoints sliding partners while the side
  coil visibly shortens;
- ~0:38  THE frame: top list freshly permuted, segment space and cords
  pixel-identical to five seconds earlier;
- ~0:46  springs pulling matched pairs together, two masks re-inflated and
  visibly improved.

Scene-specific cautions: cords are braids of three thin strands after shot 3 —
keep the braid subtle at wide framing; endpoints slide along the disc, never
jump; the ∅ threads are the thinnest stroke the renderer keeps visible.
Two captions maximum in the whole scene.
```

### Prompt 2 — `masked_attention`

```
Implement Storyboard B (masked_attention, ~42 s). Delete the previous
masked-attention scene; this replaces it.

Freeze-frames:
- ~0:05  the muddy drink: a braid dominated by hair-thin cool strands, the orb's
  fill visibly mud;
- ~0:17  the conservation beat: outside strands cut, surviving strands
  mid-thickening, beam total width unchanged — this is the frame the scene
  exists for;
- ~0:24  second breath: stencil hole visibly tighter than the first, orb warmer;
- ~0:29  the failsafe: stencil dissolved for one layer, open drink, recovery;
- ~0:40  the assembled softmax( QK^T + M ) V with the stencil shrunk into +M.

Cautions: strand thickness is the entire argument — animate width, not
opacity, for the renormalization; the only text on the stencil is −∞; the
measured 0.20 → 0.59 line may appear only in the final residue second, muted.
```

### Prompt 3 — `query_becomes_segment`

```
Implement Storyboard C (query_becomes_segment, ~35 s).

Freeze-frames:
- ~0:04  orbs pre-decoder already casting rough fields (born with opinions);
- ~0:12  two orbs contesting one duck, the single self-attention spark, one
  yielding;
- ~0:19  an orb refracting through the prism into its class tint;
- ~0:30  the three-way regroup of the SAME fields: semantic / instance /
  panoptic, nothing re-computed, only grouping changing in place.

Cautions: the decoder is nine faint gates, nothing more — no labeled blocks;
the fields are soft luminance, not outlined regions, until they crystallize.
```

### Prompt 4 — `scales_breathe`

```
Implement Storyboard D (scales_breathe, ~30 s). Must loop seamlessly.

Freeze-frames:
- ~0:04  the duckling mid-dissolve during the dive to the frosted pane;
- ~0:15  mid-breath: the orb's field visibly gaining structure coarse-to-fine;
- ~0:25  the fused slab sagging, the orb's glide slowed to a crawl;
- final  the resumed light breath, loop point invisible.

Cautions: pane depth is a real z-axis camera composition, not three rectangles
side by side; level identity is an edge tint each pane wears, never a label.
```

### Prompt 5 — `shoreline_probes`

```
Implement Storyboard E (shoreline_probes, ~32 s).

Freeze-frames:
- ~0:05  the shoreline ignited between the two settled sheets;
- ~0:11  the whole sheet on the left pan, beam dipped;
- ~0:18  sparked needles on the right pan, beam LEVEL — the proof frame;
- ~0:23  split screen: identical constellation stamped on three pairs (left),
  needles magnetized to the shoreline (right);
- ~0:30  the half-beat 112×112 lattice, then rest.

Cautions: the balance beam is the argument for unbiasedness — its leveling
must be slow and unmistakable; cold needles never disappear, they dim.
```

### Prompt 6 — integration & QA (after all five)

```
1. Swap the article's figure embeds per storyboards.md §8: video elements
   (webm + mp4 fallback, gif last resort), autoplay loop muted playsinline,
   lazy-loaded except the §4.2 hero; keep captions, reword lightly to the new
   imagery (cords, stencil, breaths, shoreline).
2. Demote meta_architecture to a static labeled diagram (single clean frame is
   fine); keep layerwise_pq as a small static chart in §4.3 or drop it — your
   call, justify in the commit message.
3. Verify each final webm < 2 MB, each scene passes the grammar checklist in
   storyboards.md §1 on three random paused frames, and the page background is
   pixel-identical to the site's around every figure.
4. Build the site, screenshot the article at desktop and mobile widths, and
   give me a short report with the screenshots.
```
