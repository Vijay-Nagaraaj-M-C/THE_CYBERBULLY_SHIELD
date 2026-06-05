<p align="center">
  <img src="Extension/extension/icons/icon128.png" alt="Cyberbully Shield Logo" width="120"/>
</p>

<h1 align="center">🛡️ The Cyberbully Shield</h1>

<p align="center">
  <strong>A multi-modal AI system that detects cyberbullying in text, images, and video — delivered as a real-time Chrome extension with a local inference backend.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Chrome_Extension-MV3-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Chrome Extension"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-dashboard">Dashboard</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📖 Overview

**The Cyberbully Shield** is an end-to-end, multi-modal cyberbullying detection system that combines deep learning with browser-level content moderation. It scans text comments, images (with OCR), and visual content across social media platforms — flagging and censoring harmful content **in real-time**, directly in your browser.

All inference runs **100% locally on your machine** — no data is ever sent to external servers. Your privacy is fully protected.

### Why This Project?

Cyberbullying remains one of the most pervasive issues on social media. Existing moderation tools are often:
- **Text-only** — missing image-based abuse (memes, screenshots of harassment)
- **Cloud-dependent** — requiring users to send private browsing data to third parties
- **Reactive** — flagging content after the damage is done

The Cyberbully Shield addresses all three: it's **multi-modal**, **local-first**, and **real-time**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔤 **Text Detection** | Classifies text comments, posts, and replies for cyberbullying |
| 🖼️ **Image Analysis** | Processes images via ResNet50 + EasyOCR to detect visual harassment |
| 🔀 **Multi-Modal Fusion** | Cross-attention mechanism fuses text and vision signals for superior accuracy |
| 🌐 **Chrome Extension** | Manifest V3 extension scans pages in real-time (Twitter/X, YouTube, Reddit, Facebook, Instagram, and more) |
| 🔴 **Live Censoring** | Detected content is blurred with a confidence badge — click to reveal |
| 📊 **Analytics Dashboard** | Web dashboard with live stats, timeline charts, and a searchable detection log |
| 🔒 **Privacy-First** | All inference runs locally via FastAPI on `localhost:5000` — zero data leaves your machine |
| ⚡ **4GB VRAM Optimized** | Mixed-precision training, gradient accumulation, and small batch sizes for consumer GPUs |

---

## 🧠 Architecture

The core of the system is **CyberbullyShieldFusion** — a custom multi-modal neural network that fuses textual and visual understanding through cross-attention.

```
                    ┌─────────────────────┐
                    │    Input Content     │
                    │  (Text and/or Image) │
                    └────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌─────────────────┐          ┌─────────────────────┐
   │   DistilBERT     │          │     ResNet50         │
   │ (Text Encoder)   │          │  (Vision Encoder)    │
   │ → (B, seq, 768)  │          │  → (B, 2048, 7, 7)  │
   └────────┬────────┘          └──────────┬──────────┘
            │                              │
            │                    ┌─────────▼──────────┐
            │                    │  Linear Projection  │
            │                    │  2048 → 768         │
            │                    │  → (B, 49, 768)     │
            │                    └─────────┬──────────┘
            │                              │
            └──────────┬───────────────────┘
                       ▼
            ┌─────────────────────┐
            │   Cross-Attention    │
            │  Q: Vision (49×768) │
            │  K,V: Text (seq×768)│
            │  8 attention heads   │
            └────────┬────────────┘
                     ▼
            ┌─────────────────────┐
            │   Mean Pooling       │
            │   → (B, 768)         │
            └────────┬────────────┘
                     ▼
            ┌─────────────────────┐
            │ Feed-Forward Classifier│
            │ 768 → 256 → 1       │
            │ (BCEWithLogitsLoss)  │
            └────────┬────────────┘
                     ▼
            ┌─────────────────────┐
            │  Cyberbullying /     │
            │  Safe (Binary)       │
            └─────────────────────┘
```

### Training Strategy

The model uses a **2-phase training** approach:

1. **Phase 1 (Epochs 1–3):** Frozen DistilBERT and ResNet50 backbones — only the cross-attention and classifier layers are trained with a learning rate of `2e-4`.
2. **Phase 2 (Epochs 4–10):** The top layers of both encoders are unfrozen for fine-tuning at a reduced learning rate of `5e-6` to prevent catastrophic forgetting.

Additional optimizations include mixed-precision (`torch.cuda.amp`), gradient accumulation (effective batch size of 32), gradient clipping, linear warmup scheduling, and early stopping.

### End-to-End Pipeline

```
User browses social media
        ↓
Content Script (content.js) scans visible text elements
        ↓
Sends text batch to local backend → POST /api/predict_batch
        ↓
Backend runs inference with CyberbullyShieldFusion model
(DistilBERT text → ResNet50 image → Cross-Attention → Classifier)
        ↓
Returns {is_bullying, confidence} for each item
        ↓
Content Script applies blur overlay + badge on flagged items
        ↓
Detection logged to SQLite → viewable on Dashboard
```

---

## 📂 Folder Structure

```
THE_CYBERBULLY_SHIELD/
├── Model/
│   ├── src/
│   │   ├── model.py                # CyberbullyShieldFusion architecture
│   │   ├── train.py                # Training script (2-phase, mixed precision)
│   │   └── data_ingestion.py       # Dataset loading, OCR, preprocessing
│   ├── checkpoints/
│   │   └── best_shield_model.pth   # Trained weights (~376 MB) — see Download section
│   └── notebooks/
│       ├── 01_evaluation_metrics.ipynb   # Model evaluation & metrics
│       └── 02_model_testing.ipynb        # Inference testing
├── Datasets/                       # Training datasets (excluded from repo — see Dataset Setup)
├── Extension/
│   ├── backend/
│   │   ├── server.py               # FastAPI server (localhost:5000)
│   │   ├── inference.py            # Model inference wrapper
│   │   ├── database.py             # SQLite detection log
│   │   ├── requirements.txt        # Python dependencies
│   │   └── dashboard/
│   │       └── index.html          # Web dashboard UI
│   └── extension/
│       ├── manifest.json           # Chrome Extension Manifest V3
│       ├── content.js              # Page scanner + overlay logic
│       ├── content.css             # Blur/badge overlay styles
│       ├── background.js           # Service worker
│       ├── popup.html              # Extension popup UI
│       ├── popup.js                # Popup logic
│       ├── popup.css               # Popup styles
│       └── icons/                  # Extension icons (16, 48, 128px)
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Deep Learning** | Python, PyTorch 2.0+, HuggingFace Transformers, DistilBERT, ResNet50, EasyOCR |
| **Backend API** | FastAPI, Uvicorn, SQLite, Pydantic |
| **Browser Extension** | JavaScript, Chrome Extension Manifest V3, HTML/CSS |
| **Training Infra** | Mixed-Precision (AMP), Gradient Accumulation, AdamW, Linear Warmup |
| **Data Processing** | Pandas, scikit-learn, Pillow, EasyOCR |

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or higher |
| **PyTorch** | 2.0+ (CUDA recommended, CPU supported) |
| **Browser** | Google Chrome, Microsoft Edge, or any Chromium-based browser |
| **GPU (optional)** | NVIDIA GPU with ≥4GB VRAM for inference; CPU fallback is supported |

### 1. Clone the Repository

```bash
git clone https://github.com/Vijay-Nagaraaj-M-C/THE_CYBERBULLY_SHIELD.git
cd THE_CYBERBULLY_SHIELD
```

### 2. Install Python Dependencies

```bash
cd Extension/backend
pip install -r requirements.txt
```

> **Note:** If you already have PyTorch, torchvision, and transformers installed from model training, you primarily need:
> ```bash
> pip install fastapi uvicorn
> ```

### 3. Download the Model Weights

The trained model checkpoint (`best_shield_model.pth`, ~376 MB) is hosted separately due to GitHub's file size limits. See the **[Download the Model](#-download-the-model)** section below for full instructions.

### 4. Start the FastAPI Backend

```bash
cd Extension/backend
python server.py
```

You should see:

```
============================================================
  THE CYBERBULLY SHIELD — Local Inference Server
  Dashboard: http://localhost:5000/dashboard
  API:       http://localhost:5000/api/health
============================================================
[ShieldPredictor] Device: cuda
[ShieldPredictor] Loading model from ...\best_shield_model.pth ...
[ShieldPredictor] Model loaded successfully.
INFO:     Uvicorn running on http://0.0.0.0:5000
```

**Verify it's running:**
- http://localhost:5000/api/health → `{"status":"online","model_loaded":true}`
- http://localhost:5000/dashboard → Opens the analytics dashboard

### 5. Load the Chrome Extension

1. Open **Chrome** (or Edge)
2. Navigate to `chrome://extensions/` (or `edge://extensions/`)
3. Enable **"Developer mode"** (toggle in the top-right corner)
4. Click **"Load unpacked"**
5. Select the folder: `THE_CYBERBULLY_SHIELD/Extension/extension`
6. The 🛡️ **Cyberbully Shield** icon should now appear in your browser toolbar

---

## 📦 Download the Model

The trained model checkpoint (~376 MB) exceeds GitHub's 100 MB file limit and is hosted on **HuggingFace Hub**.

### Option A: Direct Download (Easiest)

1. Go to: **[Cyberbully-Shield-Model on HuggingFace](https://huggingface.co/Vijay-Nagaraaj-M-C/Cyberbully-Shield-Model)**
2. Click on `cyberbully-shield-model.pth` → **Download**
3. Rename the file to `best_shield_model.pth`
4. Place it in `Model/checkpoints/`

### Option B: HuggingFace CLI

```bash
pip install huggingface_hub

# Download the checkpoint
hf download Vijay-Nagaraaj-M-C/Cyberbully-Shield-Model \
    cyberbully-shield-model.pth \
    --local-dir Model/checkpoints

# Rename to match what the backend expects
cd Model/checkpoints
mv cyberbully-shield-model.pth best_shield_model.pth
```

> 🔗 **Model Card:** [https://huggingface.co/Vijay-Nagaraaj-M-C/Cyberbully-Shield-Model](https://huggingface.co/Vijay-Nagaraaj-M-C/Cyberbully-Shield-Model)

### After Downloading

Ensure the file is placed and named correctly:

```
THE_CYBERBULLY_SHIELD/
└── Model/
    └── checkpoints/
        └── best_shield_model.pth   ← must be this exact name
```

> **Note:** The backend server (`inference.py`) expects the file to be named `best_shield_model.pth`. If you skip the rename step, the server will fail to start.

---

## 📊 Dataset Setup

The training datasets are sourced from **Kaggle** and are **not included** in this repository (excluded via `.gitignore`). You must download them manually if you wish to retrain the model.

### Required Datasets

| # | Dataset | Kaggle Link | Type |
|---|---------|-------------|------|
| 1 | **Cyberbullying Tweets** | [soorajtomar/cyberbullying-tweets](https://www.kaggle.com/datasets/soorajtomar/cyberbullying-tweets) | Text-only |
| 2 | **Cyberbullying Dataset** | [saurabhshahane/cyberbullying-dataset](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset) | Text-only |
| 3 | **Multimodal Cyberbullying** | [studentramya/multimodal-cyberbullying](https://www.kaggle.com/datasets/studentramya/multimodal-cyberbullying) | Text + Images |

> **Note:** If the exact links above do not work, search Kaggle for similar cyberbullying / hate speech classification datasets with compatible column schemas (see `data_ingestion.py` for expected formats).

### Download & Placement Instructions

1. Download each dataset from Kaggle
2. Place them in the `Datasets/` folder with this structure:

```
THE_CYBERBULLY_SHIELD/
└── Datasets/
    ├── cyberbullying-tweets.csv
    ├── cyberbullying-dataset/
    │   └── *.csv
    └── multimodal-cyberbullying/
        ├── cyberbully.csv
        └── bully_data/
            └── *.jpg / *.png
```

3. Verify loading by running:

```bash
cd Model/src
python -c "from data_ingestion import load_all_datasets; load_all_datasets()"
```

### Expected Column Schemas

| Dataset | Expected Columns | Label Mapping |
|---------|-----------------|---------------|
| cyberbullying-tweets.csv | `Text`, `CB_Label` | `1.0` = bullying, `0.0` = safe |
| cyberbullying-dataset/*.csv | `Text`, `oh_label` | `1.0` = bullying, `0.0` = safe |
| multimodal-cyberbullying | `Img-Name`, `Img-Text`, `Img-Label` | `"bully"` = 1.0, else = 0.0 |

---

## 🏋️ Training

To retrain the model from scratch (requires datasets — see above):

```bash
cd Model/src
python train.py
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 (with early stopping, patience=3) |
| Batch Size | 4 (effective 32 via gradient accumulation) |
| Phase 1 LR | `2e-4` (frozen encoders) |
| Phase 2 LR | `5e-6` (unfrozen top layers, from epoch 4) |
| Optimizer | AdamW (weight decay `1e-2`) |
| Scheduler | Linear warmup (10% of total steps) |
| Precision | Mixed precision (`torch.cuda.amp`) |
| Loss | `BCEWithLogitsLoss` |

The best checkpoint is saved to `Model/checkpoints/best_shield_model.pth` based on validation loss.

---

## 🖥️ Usage

### End-to-End Workflow

```
1. Start the backend     →  python Extension/backend/server.py
2. Load the extension     →  chrome://extensions/ → Load unpacked
3. Browse the web         →  Visit Twitter, YouTube, Reddit, etc.
4. See live protection    →  Bullying content is blurred + badged
5. Review detections      →  http://localhost:5000/dashboard
```

### Extension Controls

Click the 🛡️ icon in your browser toolbar to open the popup:

| Control | Description |
|---------|-------------|
| **Protection Toggle** | Turn scanning ON/OFF |
| **Sensitivity Slider** | Adjust the confidence threshold (30%–90%). Lower = more aggressive detection |
| **Scan Now** | Force an immediate re-scan of the current page |
| **Dashboard** | Opens the analytics dashboard in a new tab |
| **Status Indicator** | 🟢 Backend running · 🔴 Backend offline |

### Supported Platforms

The content script includes built-in selectors for:

- **Twitter / X** — Tweet text
- **YouTube** — Comment text
- **Reddit** — Comments and posts
- **Facebook** — Post content
- **Instagram** — Captions
- **Any page** — Falls back to generic selectors (`article`, `comment`, `post` elements)

---

## 📊 Dashboard

Access the analytics dashboard at: **http://localhost:5000/dashboard**

The dashboard provides:

- **Live Stats** — Total scans, threats detected, safe content count, threat rate percentage
- **Timeline Chart** — Detections plotted over the last 24 hours
- **Classification Pie** — Bullying vs. Safe content distribution
- **Detection Log** — Scrollable table of all flagged content
  - Censored text is blurred — **hover to reveal**
  - Shows timestamp, source URL, and confidence score
- **Clear History** — Wipe all detection logs with a single click
- Auto-refreshes every 10 seconds

---

## 🔌 API Reference

The FastAPI backend exposes the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | `GET` | Check if the backend is online and model is loaded |
| `/api/predict` | `POST` | Single text/image prediction |
| `/api/predict_batch` | `POST` | Batch text predictions (used by the extension) |
| `/api/stats` | `GET` | Aggregate detection statistics |
| `/api/history?limit=N` | `GET` | Recent detection log entries |
| `/api/clear` | `POST` | Clear all detection history |
| `/dashboard` | `GET` | Web dashboard UI |

### Example Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are so ugly nobody likes you", "source_url": "https://example.com"}'
```

```json
{
  "is_bullying": true,
  "confidence": 0.9312,
  "label": "Cyberbullying"
}
```

---

## ❓ Troubleshooting

<details>
<summary><strong>Backend won't start — <code>ModuleNotFoundError</code></strong></summary>

```bash
pip install fastapi uvicorn torch torchvision transformers Pillow
```
</details>

<details>
<summary><strong>Extension shows "Backend offline"</strong></summary>

- Ensure `python server.py` is running in a terminal
- Verify: http://localhost:5000/api/health
- Check that no firewall is blocking port 5000
</details>

<details>
<summary><strong>No content is being scanned</strong></summary>

- Click the extension icon and verify **Protection** is ON
- Click **Scan Now** to force a scan
- The content script only scans text elements with ≥ 8 characters
</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

- Close other GPU-intensive applications
- Force CPU mode by editing `inference.py`:
  ```python
  self.device = torch.device('cpu')
  ```
</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** this repository
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Commit your changes:** `git commit -m "Add amazing feature"`
4. **Push to the branch:** `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution

- 🌍 Multi-language support (currently English-only)
- 🎬 Video frame analysis pipeline
- 📱 Firefox / Safari extension ports
- 📈 Model accuracy improvements with additional datasets
- 🎨 Dashboard UI/UX enhancements
- 🧪 Unit tests and CI/CD pipeline

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **[Kaggle](https://www.kaggle.com/)** — Training datasets for cyberbullying classification
- **[HuggingFace](https://huggingface.co/)** — Transformers library and model hosting
- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance Python API framework
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — Optical character recognition for image text extraction
- **[DistilBERT](https://huggingface.co/distilbert-base-uncased)** — Efficient text encoder
- **[ResNet50](https://pytorch.org/vision/stable/models.html)** — Pre-trained vision backbone

---

<p align="center">
  <strong>Built with ❤️ to make the internet a safer place.</strong>
</p>

<p align="center">
  <a href="#️-the-cyberbully-shield">Back to Top ↑</a>
</p>
