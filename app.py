"""
app.py — Gradio demo for Hugging Face Spaces (free tier, native Gradio SDK).

No separate backend needed — Gradio handles both the UI and inference
in one process, loading the model once at startup.
"""
try:
    import spaces  # must be imported before torch/any CUDA-related package
except ImportError:
    pass  # not running on a Spaces environment that requires this

import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "fusion_model_final.pt"  # must sit in the same repo, committed via Git LFS
TEXT_MODEL_NAME = "distilbert-base-multilingual-cased"
LABELS = ["compliant", "non-compliant"]


class FusionModel(nn.Module):
    """Must match training architecture exactly for state_dict to load correctly."""
    def __init__(self, img_backbone, text_dim=768, img_dim=512, shared_dim=256, hidden=256):
        super().__init__()
        self.img_backbone = img_backbone
        self.img_backbone.fc = nn.Identity()
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.img_proj = nn.Linear(img_dim, shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 4, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2)
        )

    def forward(self, imgs, text_emb):
        img_feat = self.img_backbone(imgs)
        t = self.text_proj(text_emb)
        v = self.img_proj(img_feat)
        interaction = torch.cat([t, v, torch.abs(t - v), t * v], dim=1)
        return self.classifier(interaction)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading text encoder...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME).eval()  # stays on CPU until moved inside the GPU-decorated call
for p in text_encoder.parameters():
    p.requires_grad = False

print("Loading fusion model...")
img_backbone = models.resnet18(weights=None)
fusion_model = FusionModel(img_backbone)
fusion_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
fusion_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def _predict_impl(image, designation, description):
    if image is None or not designation:
        return "Please provide both an image and a title.", None

    # Move models to GPU here — ZeroGPU only attaches a real GPU inside
    # a @spaces.GPU-decorated call, not at module import time.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.to(device)
    fusion_model.to(device)

    pil_image = Image.fromarray(image).convert("RGB")
    text = f"{designation} {description or ''}".strip()

    img_tensor = img_transform(pil_image).unsqueeze(0).to(device)
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt").to(device)
    text_emb = text_encoder(**enc).last_hidden_state[:, 0, :]

    logits = fusion_model(img_tensor, text_emb)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    result_label = f"{'✅ Compliant' if probs[0] > probs[1] else '⚠️ Non-compliant'}"
    confidence_dict = {"compliant": float(probs[0]), "non-compliant": float(probs[1])}

    return result_label, confidence_dict


# ZeroGPU requires inference to happen inside a function decorated with @spaces.GPU
# so Hugging Face knows when to allocate a GPU to this process. If the `spaces`
# package isn't available (e.g. running locally, not on a ZeroGPU Space), fall
# back to calling the function directly with no decoration.
try:
    predict = spaces.GPU(_predict_impl)
except NameError:
    predict = _predict_impl


with gr.Blocks(title="Multimodal Content Moderation Demo") as demo:
    gr.Markdown("# 🛍️ Multimodal Content Moderation Demo")
    gr.Markdown(
        "Upload a product photo and its listing text. The model predicts whether "
        "they genuinely correspond, or whether the pairing looks inconsistent — "
        "a proxy signal for mislabeled or recycled-image listings."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Product image")
            designation_input = gr.Textbox(label="Product title", placeholder="e.g. Piscine hors sol gré 915 x 470 cm")
            description_input = gr.Textbox(label="Description (optional)", lines=3)
            submit_btn = gr.Button("Check consistency", variant="primary")
        with gr.Column():
            result_output = gr.Textbox(label="Result")
            confidence_output = gr.Label(label="Confidence")

    submit_btn.click(
        fn=predict,
        inputs=[image_input, designation_input, description_input],
        outputs=[result_output, confidence_output],
    )

    gr.Markdown(
        "⚠️ Trained on a synthetic mismatch-detection proxy task (Rakuten France "
        "product catalog with artificially swapped images), not real content-policy "
        "violation data. English text is out-of-distribution — the model was "
        "trained only on French listings."
    )

if __name__ == "__main__":
    demo.launch()
