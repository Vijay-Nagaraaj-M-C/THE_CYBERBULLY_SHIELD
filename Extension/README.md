# 🛡️ The Cyberbully Shield — Browser Extension

Real-time cyberbullying detection browser extension powered by the CyberbullyShieldFusion AI model.
All inference runs **locally on your machine** — no data leaves your system.

---

## 📁 Project Structure

```
THE_CYBERBULLY_SHIELD/
├── Model/                          # Trained AI model
│   ├── src/
│   │   ├── model.py                # CyberbullyShieldFusion architecture
│   │   ├── train.py                # Training script
│   │   └── data_ingestion.py       # Dataset loading & preprocessing
│   ├── checkpoints/
│   │   └── best_shield_model.pth   # Trained weights (~376 MB)
│   └── notebooks/                  # Evaluation notebooks
├── Datasets/                       # Training datasets
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
│       └── icons/                  # Extension icons
└── Documents/                      # Project report
```

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.10+** installed
- **PyTorch** with CUDA support (optional, CPU works too)
- **Google Chrome**, **Microsoft Edge**, or any Chromium-based browser

---

### Step 1: Install Backend Dependencies

Open a terminal and run:

```powershell
cd Extension/backend
pip install -r requirements.txt
```

> **Note:** If you already have PyTorch, transformers, and torchvision installed
> (which you do from model training), you mainly need `fastapi` and `uvicorn`:
>
> ```powershell
> pip install fastapi uvicorn
> ```

---

### Step 2: Start the Backend Server

```powershell
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

**Verify it's working** by opening in your browser:
- http://localhost:5000/api/health → Should show `{"status":"online","model_loaded":true}`
- http://localhost:5000/dashboard → Should show the dashboard

---

### Step 3: Load the Chrome Extension

1. Open **Chrome** (or Edge)
2. Navigate to `chrome://extensions/` (or `edge://extensions/`)
3. Enable **"Developer mode"** (toggle in top-right corner)
4. Click **"Load unpacked"**
5. Select the folder: `Extension/extension` (inside the cloned project directory)
6. The 🛡️ Cyberbully Shield extension icon should appear in your toolbar

---

### Step 4: Browse with Protection

1. Make sure the backend server is running (Step 2)
2. Visit any social media site (Twitter/X, YouTube, Reddit, Facebook, etc.)
3. The extension will automatically scan text content on the page
4. **Detected cyberbullying** will be:
   - 🔴 **Blurred/censored** with a red badge showing confidence %
   - 🔔 A toast notification appears at the bottom-right
   - 📊 Logged to the dashboard for review
5. **Click the badge** on any censored item to reveal the original content

---

## 📊 Using the Dashboard

Access the dashboard at: **http://localhost:5000/dashboard**

The dashboard shows:
- **Live Stats**: Total scans, threats detected, safe content count, threat rate
- **Timeline Chart**: Detections over the last 24 hours
- **Classification Pie**: Bullying vs Safe content distribution
- **Detection Log**: Scrollable table with all flagged content
  - Censored text is blurred — **hover to reveal**
  - Shows timestamp, source URL, confidence score

### Dashboard Controls
- **Clear History** button: Wipes all detection logs
- Auto-refreshes every 10 seconds

---

## ⚙️ Extension Controls

Click the 🛡️ icon in your browser toolbar to open the popup:

| Control | Description |
|---------|-------------|
| **Protection Toggle** | Turn scanning ON/OFF |
| **Sensitivity Slider** | Adjust the confidence threshold (30% – 90%). Lower = more aggressive detection |
| **Scan Now** | Force an immediate re-scan of the current page |
| **Dashboard** | Opens the dashboard in a new tab |
| **Status Indicator** | 🟢 Green = backend running, 🔴 Red = backend offline |

---

## 🔧 How It Works

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

### Supported Platforms
The content script has built-in selectors for:
- **Twitter / X** — Tweet text
- **YouTube** — Comment text
- **Reddit** — Comments and posts
- **Facebook** — Post content
- **Instagram** — Captions
- **Any page** — Falls back to generic selectors (article, comment, post elements)

---

## ❓ Troubleshooting

### Backend won't start
```
ModuleNotFoundError: No module named 'fastapi'
```
→ Run: `pip install fastapi uvicorn`

### Extension shows "Backend offline"
→ Make sure `python server.py` is running in a terminal
→ Check http://localhost:5000/api/health

### No content is being scanned
→ Click the extension icon and verify "Protection" is ON
→ Click "Scan Now" to force a scan
→ The content script only scans text ≥ 8 characters

### CUDA out of memory
→ The backend server loads the full multimodal model
→ Close other GPU-intensive applications
→ Or force CPU mode: edit `inference.py` line 51: `self.device = torch.device('cpu')`

---

## 🛑 Stopping the System

1. **Stop the backend**: Press `Ctrl+C` in the terminal running `server.py`
2. **Disable the extension**: Click the extension icon → toggle Protection OFF
   Or go to `chrome://extensions/` and disable it

---

## 📄 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check if backend is online |
| `/api/predict` | POST | Single text/image prediction |
| `/api/predict_batch` | POST | Batch text predictions |
| `/api/stats` | GET | Aggregate detection statistics |
| `/api/history?limit=N` | GET | Recent detection log |
| `/api/clear` | POST | Clear all detection history |
| `/dashboard` | GET | Web dashboard UI |
