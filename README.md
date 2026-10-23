# Product Listing Image–Text Consistency Experiment

A multimodal research project that compares a product image with its title/description to identify deliberately constructed image–text mismatches. The mismatch label is a proxy task, not a general content-moderation decision.

## What is included

- Notebook-based data preparation and experiment workflow
- Image/text fusion model and demo code, with a separately maintained Hugging Face Space
- Label-generation regression tests and checkpoint loading safeguards

## Hosted demo

The hosted Gradio demo runs as a separate Hugging Face Space so its deployment files remain at the Space repository root. [Open the Hugging Face demo](https://huggingface.co/spaces/iyinoluwa/moderation-demo). Its prediction is a synthetic listing-consistency signal, not a moderation decision.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

## Verify

The test suite covers mismatch-label generation. A synthetic forward pass has checked the image checkpoint/model contract. The full text-encoder and hosted UI path requires model downloads and has not been verified end to end.

## Repository map

`notebooks/` contains the experiment. `src/` contains shared code and label generation. `app.py` is the demo entry point; `tests/` covers label construction. The trained image checkpoint is stored in `models/`; the large datasets remain local in the ignored `data/` folder.

## Evidence and limitations

Label tests pass and a synthetic image-model forward pass is finite. This does not validate semantic matching quality, text-encoder downloads, or the hosted application.

The task uses synthetic/donor mismatch construction and is not a policy moderation system. Build train/validation splits before donor swaps to avoid leakage. Confirm dataset licenses, then evaluate on representative real listings and run the complete encoder/UI path before deployment.

## License

No repository license file is included. Unless the project owner adds a license, do not assume the code, data or model artifacts may be redistributed or reused. Verify third-party data and model terms separately.
