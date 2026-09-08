# Multimodal Content Moderation: Text-Image Consistency Detection

**🔗 Live Demo:** [huggingface.co/spaces/iyinoluwa/moderation-demo](https://huggingface.co/spaces/iyinoluwa/moderation-demo)

*A machine learning system that checks whether a product's photo actually matches its description — try it live using the link above.*

---

## 1. The Problem (in plain English)

When you shop online, every product listing has two parts: a **photo** and a **text description**. Most of the time these match — a photo of a blue vase, titled "Blue ceramic vase."

But sometimes they don't match, because of:
- A seller reusing an old photo for a new, different product
- Copy-paste mistakes during bulk uploads
- Deliberately misleading photos (showing something nicer than what's actually sold)

Large marketplaces (Amazon, Jumia, Etsy, etc.) need a way to **automatically flag listings where the photo and text don't agree**, so a human moderator can review them before the listing goes live to shoppers.

**The catch:** to catch this, you can't just look at the text alone (it might read perfectly fine on its own) or the photo alone (it might be a perfectly normal photo of *something*). You only catch the problem by checking whether the two **agree with each other**. That requires an AI system that looks at both at the same time — this is called a "multimodal" system, because it combines two modes of information (text + images).

## 2. The Solution This Project Offers

This project builds and trains exactly that: an AI model that takes in a product's **title + description** and its **photo**, and predicts one of two things:

- ✅ **Compliant** — the text and photo genuinely go together
- ⚠️ **Non-compliant** — the text and photo appear to be mismatched

To prove this works, three different models were built and compared:
1. A model that only looks at the **text** (to prove text alone isn't enough)
2. A model that only looks at the **image** (to prove image alone isn't enough)
3. The final **fusion model** that looks at both together (and performs dramatically better than either alone — more than double the accuracy on catching mismatches)

This comparison is the core scientific result of the project: it proves that combining both types of information is genuinely necessary, not just a nice-to-have.

## 3. A Concrete Example

Imagine a listing on an e-commerce site:

> **Title:** "Piscine hors sol gré 915 x 470 cm" (Above-ground pool)
> **Photo:** [a photo of a motorcycle helmet]

A human glancing at just the title thinks it's a normal pool listing. A human glancing at just the photo thinks it's a normal helmet photo. It's only when you look at **both together** that something is obviously wrong. This is exactly the kind of error this system is built to catch automatically, at a scale no team of human moderators could keep up with manually — a real platform receives thousands of new listings per day.

## 4. Why the Data Isn't "Real Prohibited Items"

An early goal for this project was to detect literally prohibited items (like weapons or drugs) in listings. After real investigation, it turned out **no public, freely-available dataset contains genuine examples of this** — for obvious reasons, platforms don't publish real fraud/violation data. The one academic dataset found with real examples requires a formal, gated request process and comes with legal restrictions on reuse.

So instead, this project uses a well-documented **substitute problem that requires the exact same underlying skill**: instead of "photo doesn't match a policy," it's "photo doesn't match the text" — deliberately created by taking real product listings and swapping 25% of their photos with an unrelated product's photo. This is a standard, respected technique in machine learning research when real-world labeled data isn't available, and it proves the same core capability: a model that can genuinely check agreement between text and images.

## 5. Results

| Model | Score (F1, higher is better) | What it tells us |
|---|---|---|
| Text-only | 0.30 | Text alone barely does better than guessing — expected, since text doesn't change when the photo is swapped |
| Image-only | 0.25 | Image alone also barely beats guessing — expected, since a swapped-in photo is still a perfectly normal photo of *something* |
| **Fusion (both together)** | **0.63** | More than double either baseline — proves the model is genuinely learning to compare the two |

## 6. Folder Structure — What Each File Does

This project lives across **two separate places**: a GitHub repository (for the research/training code) and a Hugging Face Space (for the live, working demo). This split is standard practice — the research code and the deployed app serve different purposes and don't need to live together.

### GitHub repo: `ecommerce_moderation`

```
ecommerce_moderation/
├── data/                              (not uploaded to GitHub — too large)
│   Contains: the raw product data (CSVs) and product images downloaded
│   from the dataset source, plus files the notebook generates while running
│   (the labeled dataset, saved model checkpoints, results).
│
├── notebooks/
│   └── moderation_pipeline_v2.ipynb
│   This is the main file. It's a Google Colab notebook containing ALL the
│   code: loading data, building the mismatch labels, training all three
│   models, and printing the results. Anyone can open this in Google Colab
│   and re-run the entire project from scratch.
│
├── src/
│   └── build_labels.py
│   A standalone version of the "label building" logic from the notebook
│   (the part that decides which listings get a swapped photo). Kept
│   separate so it can be reused or tested independently of the notebook.
│
├── README.md
│   This file — explains the whole project.
│
├── requirements.txt
│   A list of all the software packages needed to run this project
│   (e.g. PyTorch, scikit-learn). Running `pip install -r requirements.txt`
│   installs everything at once.
│
└── .gitignore
    A configuration file that tells Git (the version control tool) which
    files to NOT upload to GitHub — mainly the large data files and model
    weights, which don't belong in a code repository.
```

### Hugging Face Space: `moderation-demo` (separate repo)

```
moderation-demo/
├── app.py
│   The actual code that runs the live demo website. Loads the trained
│   model and creates the web interface (upload a photo, type a title,
│   click a button, see the result) using a tool called Gradio.
│
├── requirements.txt
│   Same idea as above, but only the packages needed to RUN the demo
│   (not to train the model).
│
└── fusion_model_final.pt
    The actual trained model — this is the "brain" that was produced by
    running the notebook. It's a large file (the model's learned
    parameters) that app.py loads when the demo starts up.
```

## 7. How the Model Was Built (Step by Step)

1. **Got the data** — downloaded ~85,000 real product listings (title, description, photo) from a public dataset released by Rakuten (a French e-commerce company) for a research competition.
2. **Created the "mismatch" labels** — since no real mismatch data exists publicly, 25% of listings had their photo swapped with an unrelated product's photo, and everything was recorded so we know exactly which listings are genuine vs. swapped.
3. **Built and tested a text-only model** — proved text alone can't solve this (as expected).
4. **Built and tested an image-only model** — proved image alone can't solve this either (as expected).
5. **Built the fusion model** — combined both, using two AI building blocks: DistilBERT (a language-understanding model) for text, and ResNet18 (an image-understanding model) for photos. It compares the two directly, and dramatically outperforms both single-mode models.
6. **Trained everything on Google Colab** (a free, cloud-based platform Google provides for running Python code with GPU acceleration, which speeds up AI training significantly).
7. **Deployed the winning (fusion) model** to Hugging Face Spaces, a free hosting service specifically built for sharing AI demos publicly.

## 8. Deployment: How It Went Live

1. Created a free account on [Hugging Face](https://huggingface.co) (a platform for hosting and sharing AI models/demos)
2. Created a new "Space" (Hugging Face's term for a hosted demo app) using the **Gradio** framework — a tool for quickly building simple web interfaces for AI models, no web-development experience required
3. Uploaded three things to that Space: the app code (`app.py`), the list of required software (`requirements.txt`), and the trained model file (`fusion_model_final.pt`)
4. Hugging Face automatically builds and hosts the app — no server management, no cost, and it's instantly live at a public web address
5. Fixed a couple of platform-specific compatibility issues along the way (e.g. how the app requests temporary GPU access from Hugging Face's free tier)

## 9. How to Use It (No Technical Knowledge Required)

1. Open the live demo: **[huggingface.co/spaces/iyinoluwa/moderation-demo](https://huggingface.co/spaces/iyinoluwa/moderation-demo)**
2. Upload any product photo (a bag, shoe, book, anything)
3. Type a product title in the text box
4. Click "Check consistency"
5. The app tells you whether it thinks the photo and text match, with a confidence percentage

**Try this to see it work:** upload a photo of one item, but type a title describing a *completely different* item. The model should flag it as "non-compliant." Then try a matching photo/title pair — it should say "compliant."

**Note:** the model was trained only on French-language product listings, so it performs best with French text — English titles are outside what it learned from, which is a known and documented limitation.

## 10. Honest Limitations

- This demonstrates a proxy capability (detecting artificially created mismatches), not real content-policy violations, since that data isn't publicly available
- Trained on a subset of the full dataset (20,000 of 85,000 listings) due to free-tier computing constraints
- Performs best on French text
- Weakest at distinguishing mismatches within visually similar/broad categories (e.g. two different home/garden items), and strongest across very different categories (e.g. a book photo vs. a toy photo)

## 11. Reproducing This Project Yourself

1. Download the dataset from [ENS Challenge Data](https://challengedata.ens.fr/challenges/35)
2. Upload the data files to your own Google Drive
3. Open `notebooks/moderation_pipeline_v2.ipynb` in Google Colab
4. Set the runtime to use a free GPU (Runtime → Change runtime type → T4 GPU)
5. Run all the cells from top to bottom
6. The trained model and results will save automatically to your Drive
