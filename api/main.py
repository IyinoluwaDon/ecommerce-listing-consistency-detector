"""
main.py — FastAPI inference service for the text-image consistency model.

Loads the trained fusion model (text_proj + img_proj + classifier weights,
plus the ResNet18 image backbone) and exposes a single endpoint that accepts
a product image + title/description and returns a compliant/non-compliant
prediction with confidence.

Run locally:
    uvicorn main:app --reload --port 8000

Then POST to http://localhost:8000/predict with multipart form data:
    - image: file
    - designation: str
    - description: str (optional)
"""
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "fusion_model_final.pt"  # place alongside this file, or update path
TEXT_MODEL_NAME = "distilbert-base-multilingual-cased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionModel(nn.Module):
    """Must match the architecture used in training exactly, or state_dict won't load."""
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


# --- Load models once at startup ---
print("Loading text encoder...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device).eval()
for p in text_encoder.parameters():
    p.requires_grad = False

print("Loading fusion model...")
img_backbone = models.resnet18(weights=None)  # weights come from the checkpoint, not ImageNet, at inference time
fusion_model = FusionModel(img_backbone).to(device)
fusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
fusion_model.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABELS = ["compliant", "non-compliant"]

app = FastAPI(title="Multimodal Content Moderation API")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    compliant_prob: float
    non_compliant_prob: float


@torch.no_grad()
def run_inference(image: Image.Image, text: str) -> PredictionResponse:
    img_tensor = img_transform(image).unsqueeze(0).to(device)

    enc = tokenizer(text, truncation=True, padding="max_length",
                     max_length=64, return_tensors="pt").to(device)
    text_out = text_encoder(**enc)
    text_emb = text_out.last_hidden_state[:, 0, :]  # CLS token

    logits = fusion_model(img_tensor, text_emb)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(probs.argmax())
    return PredictionResponse(
        label=LABELS[pred_idx],
        confidence=float(probs[pred_idx]),
        compliant_prob=float(probs[0]),
        non_compliant_prob=float(probs[1]),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    designation: str = Form(...),
    description: str = Form(""),
):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = f"{designation} {description}".strip()
    return run_inference(pil_image, text)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}
