# E-commerce Listing Consistency Project

This project compares a product image with its title and description to study synthetic image-text mismatches. Its labels are experimental proxy classes, not moderation decisions or a general content-safety classifier.

## Components

- `github-main/` — the canonical GitHub repository checkout on `main`. It contains the research workflow, source, tests, and local data. Large data files are ignored by Git.
- `huggingface-space/` — the hosted Gradio demo checkout, maintained in its own Hugging Face Space repository so that the Space's `app.py`, `requirements.txt`, model, and README configuration stay at that repository's root.
- `github-api-branch/` — an optional local FastAPI experiment on another branch of the same GitHub repository. It is not required to run or deploy the canonical project and is not included when pushing `github-main`'s `main` branch.

The Space checkpoint is byte-for-byte identical to the main checkout's model checkpoint. GitHub stores the project source; Hugging Face builds and hosts the demo.

## Run locally

Use `github-main/README.md` for installation, tests, and the research workflow. To run the hosted demo locally, enter `huggingface-space/` and follow its README. Run Git commands from the relevant component folder so commits go to the intended repository. The suite folder itself is not a Git repository.

## Data and limitations

The local datasets are large and remain under `github-main/data/`, ignored by Git. This project has not established real-world semantic matching quality or moderation performance; evaluate it with a representative, license-cleared dataset before relying on predictions.
