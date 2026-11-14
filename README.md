# 🎨 Neural Style Transfer — Web Application

A deep learning web app that transforms photos into artistic masterpieces using **AdaIN-based fast neural style transfer** with VGG-19, trained on authentic **WikiArt** master paintings.

**Author:** Vaibhav Sharma — B.Tech CSE / AI & ML

---

## ⚡ Quick Run (Easiest)

On Windows, simply double-click:
```
run_app.bat
```
This automatically starts the server and opens `http://localhost:5000` in your web browser.

For full detailed steps, see [HOW_TO_RUN.md](file:///c:/Users/Vaibhav%20Sharma/Downloads/Neural-Style-Transfer1/HOW_TO_RUN.md).

---

## 💻 Manual Terminal Start

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Run the web app
python server.py

# 3. Open in browser
# http://localhost:5000
```

> **Note:** The high-resolution model (`adain_decoder.pth`) is already trained on the curated WikiArt dataset and loaded!

---

## 📁 Project Structure

```
Neural-Style-Transfer1/
│
├── run_app.bat         ← 🚀 1-Click launcher (starts server & opens browser)
├── HOW_TO_RUN.md       ← 📖 Complete step-by-step user guide
├── server.py           ← Flask backend (API + serves HTML)
├── adain_model.py      ← VGG-19 Encoder + AdaIN Decoder architecture
├── utils.py            ← AdaIN & mean/std helpers
├── train_fast.py       ← High-res fine-tuning script
├── adain_decoder.pth   ← High-Resolution WikiArt model weights
│
├── templates/
│   └── index.html      ← Modern web UI (Jinja2)
│
├── static/
│   ├── css/style.css   ← Glassmorphism dark styling
│   └── js/app.js       ← Frontend logic (drag & drop, live progress, previews)
│
├── data/
│   ├── content/        ← 48 Content photos (portraits, landscapes, cities)
│   └── style/          ← 47 Style artworks (17 genuine WikiArt masterpieces)
│
├── output/             ← Saved stylized images
├── uploads/            ← Temporary upload storage (auto-cleaned)
│
├── requirements.txt
└── README.md
```

---

## ✨ Features

- **⚡ Fast Real-Time Inference:** Generates museum-grade artistic outputs in ~1.5–3 seconds using fine-tuned AdaIN decoder weights.
- **↔ Interactive Split-Screen Slider:** Swipe or drag a neon divider to compare Original Photo vs AI Synthesized Artwork in real time.
- **🎨 Quick Masterpiece Inspirations:** 1-click preset bar featuring genuine masterworks from Van Gogh, Klimt, Monet, Munch, Kandinsky, Dalí, Pollock, and Picasso.
- **📸 Live Webcam Selfie Capture:** Snap a photo directly from your laptop/webcam and immediately stylize your portrait.
- **🌈 Natural Color Preservation:** Toggle switch to retain natural skin tones and scene colors while embedding painterly brushstroke textures.
- **🛠️ Art Lab Fine-Tuning:** Live sliders for Brightness, Contrast, Saturation, and Warmth with instant visual feedback and canvas-rendered downloads.
- **📋 1-Click Clipboard Copy & Fullscreen Zoom:** Instantly copy high-resolution artwork to your clipboard or open full-screen lightbox inspection.
- **🖼️ Creations Gallery:** Automatic visual archive of synthesized artworks ready for download.

---

## ⚡ How It Works

### Fast Mode (AdaIN)
1. **Encode** content & style images through frozen VGG-19 (up to `relu4_1`).
2. **AdaIN** aligns content feature statistics to style statistics.
3. **Decode** transformed features back to pixel space via the trained decoder.
4. Generates rich artistic textures in **~1–2 seconds**.

```
t = σ(style) × (f_content − μ(f_content)) / σ(f_content) + μ(style)
```

---

## 📊 Training & Loss Optimization Progression

| Stage | Dataset | Resolution | Epochs | Best Loss | Result Quality |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Initial Training** | Synthetic presets | 128×128 | 10 | 16.16 | Coarse patterns, initial structure |
| **High-Res Fine-Tuning** | Baseline dataset | 256×256 | 15 | 8.31 | Sharper spatial edges |
| **WikiArt Masterpieces** | **48 Content + 47 WikiArt Styles** | **256×256** | **15** | **11.9384 🏆** | **Authentic master brushwork, real oil textures, rich color palettes** |
