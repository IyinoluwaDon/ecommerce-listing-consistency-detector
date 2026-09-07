# Multimodal Content Moderation: Text-Image Consistency Detection

**🔗 Live Demo:** [huggingface.co/spaces/iyinoluwa/moderation-demo](https://huggingface.co/spaces/iyinoluwa/moderation-demo)

## Problem

E-commerce platforms need to catch listings where the product photo doesn't
match the title/description — a common real-world signal of recycled stock
photos, mislabeled uploads, or bait-and-switch listings. This is one practical
input into a broader content-moderation pipeline (alongside policy/category
checks, seller history, etc.).

**Why this needs both modalities:** a listing's text can look perfectly
normal in isolation, and its image can look perfectly normal in isolation —
the problem is only visible when you check whether the two *agree* with each
other. Neither a text-only nor an image-only model has any way to detect
that; only a model that reasons over both together can.

## Why a proxy task, and why it's still a valid demonstration

No public dataset contains genuine content-moderation-violation labels (real
platforms don't release that data, for obvious reasons). We evaluated
several alternatives before settling on the current design:

- **Real prohibited-item categories** (weapons, drugs) — the only academic
  dataset found with genuine categories like this
  ([Scientific Reports, 2025](https://www.nature.com/articles/s41598-025-07043-0))
  requires a gated access request and a non-commercial data-use agreement, so
  it isn't freely available for a self-contained portfolio project.
- **Arbitrary category flagging** on this dataset — rejected. The Rakuten
  catalog used here (books, toys, furniture, games) contains no genuinely
  restricted categories; manually keyword-searching the full dataset for
  weapon/drug-adjacent terms turned up only false positives (e.g. "gunpowder
  tea", a kitchen "knife" scraper, a toy water pistol) — confirming there is
  no real signal to exploit here.
- **Text-image mismatch detection (used here)** — a defensible, well-labeled
  proxy for a real moderation signal. Ground truth is constructed by
  deliberately swapping ~25% of images with a donor from a semantically
  distant category, leaving text untouched. This makes the task genuinely
  require cross-modal reasoning: text alone cannot know its image is wrong,
  and vice versa.

## Dataset

[Rakuten France Multimodal Product Data Challenge](https://challengedata.ens.fr/challenges/35)
— 84,916 product listings with title (`designation`), optional description,
and a linked product image, across 27 category codes.

## Label construction

See `src/build_labels.py`. Summary:
- 75% of listings keep their genuine image (label = `compliant`)
- 25% have their image swapped with a listing from a different, semantically
  distant category group (label = `non-compliant`) — grouped into
  `books_media`, `toys_games_figures`, `home_furniture_garden` based on
  manual inspection of sample listings per category code
- Swap rate is verified balanced across all three groups (~25% each), so the
  label isn't confounded with any single category

## Models & ablation

All three models train and evaluate on an **identical fixed subsample**
(20,000 rows, 80/20 stratified split, seed=42) — necessary for a fair,
apples-to-apples comparison.

| Model | Description | F1 (non-compliant) |
|---|---|---|
| Text-only | TF-IDF (1-2 grams) + Logistic Regression | 0.298 |
| Image-only | Pretrained ResNet18, fine-tuned, 4 epochs | 0.253 |
| **Fusion** | See architecture below, 4 epochs | **0.630** |

**Result:** both unimodal baselines sit near the floor expected of a model
with no real signal on this task (by construction — text is untouched by the
swap, and a swapped-in image is itself a perfectly normal photo of *some*
product). Fusion more than doubles the F1 of either baseline, with training
loss decreasing steadily (0.66 → 0.36) rather than overfitting noise —
strong evidence the model is learning genuine cross-modal alignment, not a
spurious shortcut.

### Fusion architecture

Detecting a mismatch is a *comparison* task, not a "predict from a pile of
features" task — naively concatenating two embeddings gives a classifier no
explicit signal about how the two relate. Instead:

1. Text: frozen `distilbert-base-multilingual-cased`, CLS token, precomputed
   once (French listings)
2. Image: fine-tuned pretrained ResNet18, 512-dim features
3. Both projected to a shared 256-dim space
4. Classifier input: `[text, image, |text − image|, text × image]` — the
   difference and product terms give the model direct access to how aligned
   or misaligned the pair is, following the standard approach used in
   NLI/sentence-matching models

## Error analysis

Confusion matrix (fusion model, validation set, n=4000):

|  | Predicted compliant | Predicted non-compliant |
|---|---|---|
| **Actual compliant** | 2561 | 436 |
| **Actual non-compliant** | 341 | 662 |

- Recall on non-compliant: 66% (662 / 1003) — catches 2 in 3 real mismatches
- Precision on non-compliant: 60% (662 / 1098) — 6 in 10 flags are genuine

**Pattern in the errors:** both false negatives (missed mismatches) and
false positives (genuine listings incorrectly flagged) are overwhelmingly
concentrated in the `home_furniture_garden` category group — pool
equipment, kitchen tools, garden hoses, cushions, all sharing one broad
label. This makes sense: that group is far more visually and topically
heterogeneous than `books_media` or `toys_games_figures`, so a swapped
image *within* it looks less obviously "wrong" than a swap between visually
distinct domains (e.g. a book cover swapped for a toy photo). The
practical implication: **the model's mismatch signal is strongest across
visually distinct domains and weakest within broad, heterogeneous
categories** — a natural target for future improvement (e.g. harder
negative mining that swaps within visually similar sub-categories, forcing
the model to learn finer-grained alignment).



- **Proxy task, not real violation data.** Results demonstrate the
  underlying multimodal capability (detecting engineered inconsistency), not
  performance on genuine policy violations, which would require labeled data
  this project does not have access to.
- **Trained on a 20K-row subsample**, not the full 84,916 rows, due to
  free-tier Colab compute constraints. Full-scale training is a
  straightforward extension.
- **Text encoder is frozen** (not fine-tuned) for training speed; end-to-end
  fine-tuning is a reasonable next step given more compute budget.
- **F1 of 0.63 on the minority class**, while a large improvement over
  baselines, still leaves room for improvement — e.g. harder negative
  mining (swapping within similar-looking categories, not just across
  distant groups), more epochs, or a larger training set.

## Repository structure

```
moderation-project/                  # this repo: training, experimentation, methodology
├── data/                          # CSVs, images, saved splits/metrics (not versioned)
├── notebooks/
│   └── moderation_pipeline.ipynb   # end-to-end pipeline, run in Colab (T4 GPU)
├── src/
│   └── build_labels.py            # standalone label-construction script
└── README.md
```

**Note:** the deployed demo lives in a **separate** repo
(`moderation-demo`, deployed on Hugging Face Spaces — see live demo link
above), since Spaces requires its own git remote. That repo only contains
the minimal inference app (`app.py`, `requirements.txt`,
`fusion_model_final.pt`) — it does not duplicate the training code or data
already documented here.

## Reproducing results

1. Download `X_train_update.csv`, `Y_train_CVw08PX.csv`, `images.zip` from
   the [ENS Challenge Data page](https://challengedata.ens.fr/challenges/35)
2. Place in Google Drive at `moderation-project/data/`
3. Open `notebooks/moderation_pipeline.ipynb` in Colab, set runtime to
   T4 GPU, run top to bottom
4. Metrics accumulate in `data/metrics.json`; model weights save every epoch
