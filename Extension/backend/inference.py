"""
inference.py — Model loading and prediction wrappers for the Cyberbully Shield.
Loads the trained CyberbullyShieldFusion model and provides fast text-only
and multimodal (text+image) prediction.
"""

import os
import sys
import re
import io
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import DistilBertTokenizer
from PIL import Image

# Add the model source directory so we can import the architecture
MODEL_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Model', 'src'))
sys.path.insert(0, MODEL_SRC)
from model import CyberbullyShieldFusion

# ── Text cleaning (same logic as data_ingestion.py) ─────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[@#]([A-Za-z0-9_]+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Image transforms (same as training) ────────────────────────────────
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ShieldPredictor:
    """
    Loads the CyberbullyShieldFusion model once and provides
    fast inference for text-only and text+image inputs.
    """

    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            checkpoint_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'Model',
                             'checkpoints', 'best_shield_model.pth'))

        print(f"[ShieldPredictor] Device: {self.device}")
        print(f"[ShieldPredictor] Loading model from {checkpoint_path} ...")

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = CyberbullyShieldFusion().to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()
        print("[ShieldPredictor] Model loaded successfully.")

    # ── core prediction ──────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, text: str, image_bytes: bytes = None) -> dict:
        """
        Predict whether content is cyberbullying.
        Returns: {"is_bullying": bool, "confidence": float, "label": str}
        """
        cleaned = clean_text(text) if text else ""

        # Tokenize text
        encoding = self.tokenizer(
            cleaned,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Process image (or use zero-tensor placeholder)
        if image_bytes:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(self.device)
            except Exception:
                image_tensor = torch.zeros((1, 3, 224, 224), device=self.device)
        else:
            image_tensor = torch.zeros((1, 3, 224, 224), device=self.device)

        # Forward pass
        logit = self.model(input_ids, attention_mask, image_tensor)
        prob = torch.sigmoid(logit).item()

        is_bullying = prob >= 0.5
        return {
            "is_bullying": is_bullying,
            "confidence": round(prob if is_bullying else 1 - prob, 4),
            "label": "Cyberbullying" if is_bullying else "Safe"
        }

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict a batch of text-only inputs for faster page scanning."""
        results = []
        # Process in mini-batches of 16
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            cleaned = [clean_text(t) for t in chunk]

            encoding = self.tokenizer(
                cleaned,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            image_tensor = torch.zeros((len(chunk), 3, 224, 224), device=self.device)

            logits = self.model(input_ids, attention_mask, image_tensor)
            probs = torch.sigmoid(logits).cpu().tolist()

            for prob in probs:
                is_bullying = prob >= 0.5
                results.append({
                    "is_bullying": is_bullying,
                    "confidence": round(prob if is_bullying else 1 - prob, 4),
                    "label": "Cyberbullying" if is_bullying else "Safe"
                })
        return results
