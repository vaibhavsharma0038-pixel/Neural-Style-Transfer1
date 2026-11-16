"""
server.py — Flask backend for Neural Style Transfer Web App
Run:  python server.py
Then open:  http://localhost:5000
"""

import os, uuid, time, threading
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from collections import namedtuple
from utils import adain, calc_mean_std

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
DECODER_PATH = os.path.join(BASE_DIR, "adain_decoder.pth")
STYLE_DIR    = os.path.join(BASE_DIR, "data", "style")
CONTENT_DIR  = os.path.join(BASE_DIR, "data", "content")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB limit

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512 if torch.cuda.is_available() else 256
FAST_MODE  = os.path.exists(DECODER_PATH)

# Global job store  { job_id: {status, progress, message, result_path} }
jobs = {}

# Model readiness flag — set to True after background warm-up completes
_model_ready = False

print(f"[server] Device    : {DEVICE}")
print(f"[server] Default Sz: {IMAGE_SIZE}")
print(f"[server] Fast mode : {FAST_MODE}")

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_tensor(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return t(img).unsqueeze(0).to(DEVICE)

def tensor_to_pil(tensor):
    img  = tensor.clone().squeeze(0).cpu()
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img  = torch.clamp(img * std + mean, 0, 1)
    return transforms.ToPILImage()(img)

def preserve_content_color(content_pil, stylized_pil):
    """Preserve natural color tones of content photo while keeping style brushwork."""
    content_resized = content_pil.resize(stylized_pil.size, Image.Resampling.LANCZOS).convert("YCbCr")
    stylized_ycbcr  = stylized_pil.convert("YCbCr")
    _, cb, cr       = content_resized.split()
    y, _, _         = stylized_ycbcr.split()
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

# ─────────────────────────────────────────────────────────────────────────────
# Models (lazy loaded, cached)
# ─────────────────────────────────────────────────────────────────────────────
_vgg = None
_encoder = None
_decoder = None
_model_ready = False   # True after background warm-up finishes

def get_vgg():
    global _vgg
    if _vgg is None:
        _vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
        for p in _vgg.parameters(): p.requires_grad_(False)
    return _vgg

def get_fast_models():
    global _encoder, _decoder
    if _encoder is None or _decoder is None:
        from adain_model import VGGEncoder, Decoder
        _encoder = VGGEncoder().to(DEVICE).eval()
        _decoder = Decoder().to(DEVICE).eval()
        _decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE, weights_only=False))
        for p in _decoder.parameters(): p.requires_grad_(False)
    return _encoder, _decoder

def _warmup_models():
    """Background thread: pre-load VGG-19 weights on startup so the first
    user request is instant. Railway health checks hit /health which waits
    for this flag before returning 200."""
    global _model_ready
    try:
        print("[warmup] Pre-loading neural models…")
        t0 = time.time()
        if FAST_MODE:
            get_fast_models()
        else:
            get_vgg()
        _model_ready = True
        print(f"[warmup] Models ready in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[warmup] ERROR during model pre-load: {e}")
        _model_ready = True  # allow health-check through so logs are visible

# Start warm-up immediately in the background
threading.Thread(target=_warmup_models, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# Fast inference
# ─────────────────────────────────────────────────────────────────────────────
def run_fast(content_path, style_path, alpha=1.0, size=IMAGE_SIZE):
    enc, dec = get_fast_models()
    ct = load_tensor(content_path, size=size)
    st = load_tensor(style_path, size=size)
    with torch.no_grad():
        cf = enc(ct)
        sf = enc(st)
        t  = alpha * adain(cf, sf) + (1 - alpha) * cf
        out = dec(t)
    return tensor_to_pil(out)

# ─────────────────────────────────────────────────────────────────────────────
# Optimization-based NST
# ─────────────────────────────────────────────────────────────────────────────
VGGOut = namedtuple("VGGOut",["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"])

def extract_features(x, vgg):
    layers = {"3":"relu1_1","8":"relu2_1","17":"relu3_1","26":"relu4_1","35":"relu5_1"}
    feats = {}
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in layers: feats[layers[name]] = x
    return VGGOut(**feats)

def gram(y):
    b,c,h,w = y.size()
    f = y.view(b, c, h*w)
    return f.bmm(f.transpose(1,2)) / (c*h*w)

def run_optimization(content_path, style_path, job_id,
                     steps=200, cw=1e5, sw=1e10, size=IMAGE_SIZE):
    vgg = get_vgg()
    ct  = load_tensor(content_path, size=size)
    st  = load_tensor(style_path, size=size)

    gen = ct.clone().requires_grad_(True)
    opt = optim.LBFGS([gen], max_iter=20, line_search_fn="strong_wolfe")
    cf  = extract_features(ct, vgg)
    sf  = extract_features(st, vgg)
    sg  = {k: gram(getattr(sf,k)) for k in sf._fields}
    wts = {"relu1_1":0.2,"relu2_1":0.4,"relu3_1":0.6,"relu4_1":0.8,"relu5_1":1.0}
    step = [0]

    def closure():
        gen.data.clamp_(0,1); opt.zero_grad()
        gf = extract_features(gen, vgg)
        cl = cw * nn.functional.mse_loss(gf.relu4_1, cf.relu4_1)
        sl = sum(wts[k]*nn.functional.mse_loss(gram(getattr(gf,k)),sg[k]) for k in sf._fields) * sw
        total = cl + sl; total.backward()
        step[0] += 1
        pct = min(int(step[0]/steps*100), 99)
        jobs[job_id]["progress"] = pct
        jobs[job_id]["message"]  = f"Optimizing… step {step[0]}/{steps}"
        return total

    for _ in range(steps // 20):
        opt.step(closure)
        if step[0] >= steps: break

    gen.data.clamp_(0,1)
    return tensor_to_pil(gen)

# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────
def process_job(job_id, content_path, style_path, alpha, steps, cw, sw, size, preserve_color):
    try:
        jobs[job_id]["status"]   = "running"
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"]  = "Loading neural representations…"

        t0 = time.time()

        # Save copies for instant before/after slider comparison
        content_preview_path = os.path.join(OUTPUT_DIR, f"{job_id}_content.png")
        style_preview_path   = os.path.join(OUTPUT_DIR, f"{job_id}_style.png")
        
        c_orig = Image.open(content_path).convert("RGB")
        s_orig = Image.open(style_path).convert("RGB")
        c_orig.save(content_preview_path)
        s_orig.save(style_preview_path)

        if FAST_MODE:
            jobs[job_id]["progress"] = 35
            jobs[job_id]["message"]  = "AdaIN feature normalization in progress…"
            result = run_fast(content_path, style_path, alpha=alpha, size=size)
            jobs[job_id]["progress"] = 85
        else:
            jobs[job_id]["message"] = "Iterative L-BFGS optimization running…"
            result = run_optimization(content_path, style_path, job_id, steps, cw, sw, size=size)

        # Apply natural color preservation if requested
        if preserve_color:
            jobs[job_id]["message"] = "Preserving content color spectrum…"
            result = preserve_content_color(c_orig, result)

        # Save stylized output
        out_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
        result.save(out_path)

        elapsed = round(time.time() - t0, 2)
        jobs[job_id].update({
            "status":       "done",
            "progress":     100,
            "message":      f"Synthesis completed in {elapsed}s",
            "result_path":  out_path,
            "content_path": content_preview_path,
            "style_path":   style_preview_path,
            "elapsed":      elapsed,
            "width":        result.width,
            "height":       result.height,
        })

    except Exception as e:
        import traceback
        jobs[job_id].update({
            "status":  "error",
            "message": str(e),
            "trace":   traceback.format_exc(),
        })
    finally:
        # Clean up temporary uploads
        for p in [content_path, style_path]:
            if p.startswith(UPLOAD_DIR):
                try: os.remove(p)
                except: pass

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    gallery  = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png") and not f.endswith("_content.png") and not f.endswith("_style.png")], reverse=True)[:18]
    return render_template("index.html",
                           fast_mode=FAST_MODE,
                           device=str(DEVICE),
                           image_size=IMAGE_SIZE,
                           gallery=gallery)

@app.route("/stylize", methods=["POST"])
def stylize():
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status":"queued","progress":0,"message":"Queued…"}

    # Pull user parameters
    alpha          = float(request.form.get("alpha", 1.0))
    steps          = int(request.form.get("steps", 200))
    cw             = float(request.form.get("content_weight", 1e5))
    sw             = float(request.form.get("style_weight",   1e10))
    size           = int(request.form.get("size", IMAGE_SIZE))
    preserve_color = request.form.get("preserve_color", "false").lower() == "true"

    # Content image (upload or data preset)
    if "content_file" in request.files and request.files["content_file"].filename:
        f = request.files["content_file"]
        content_path = os.path.join(UPLOAD_DIR, f"{job_id}_content.jpg")
        f.save(content_path)
    elif request.form.get("content_preset"):
        content_path = os.path.join(CONTENT_DIR, request.form["content_preset"])
    else:
        return jsonify({"error": "No content image provided"}), 400

    # Style image (upload or data preset)
    if "style_file" in request.files and request.files["style_file"].filename:
        f = request.files["style_file"]
        style_path = os.path.join(UPLOAD_DIR, f"{job_id}_style.jpg")
        f.save(style_path)
    elif request.form.get("style_preset"):
        style_path = os.path.join(STYLE_DIR, request.form["style_preset"])
    else:
        return jsonify({"error": "No style image provided"}), 400

    # Background worker
    t = threading.Thread(
        target=process_job,
        args=(job_id, content_path, style_path, alpha, steps, cw, sw, size, preserve_color),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route("/result/<job_id>")
def result(job_id):
    job = jobs.get(job_id, {})
    path = job.get("result_path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Result not ready"}), 404
    return send_file(path, mimetype="image/png", as_attachment=False)

@app.route("/result-content/<job_id>")
def result_content(job_id):
    path = os.path.join(OUTPUT_DIR, f"{job_id}_content.png")
    if not os.path.exists(path):
        return jsonify({"error": "Content not found"}), 404
    return send_file(path, mimetype="image/png")

@app.route("/result-style/<job_id>")
def result_style(job_id):
    path = os.path.join(OUTPUT_DIR, f"{job_id}_style.png")
    if not os.path.exists(path):
        return jsonify({"error": "Style not found"}), 404
    return send_file(path, mimetype="image/png")

@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id, {})
    path = job.get("result_path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Result not ready"}), 404
    return send_file(path, mimetype="image/png",
                     as_attachment=True, download_name=f"neural_art_{job_id}.png")

@app.route("/gallery/<filename>")
def gallery_img(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename))

@app.route("/preset/style/<filename>")
def preset_style(filename):
    return send_file(os.path.join(STYLE_DIR, filename))

@app.route("/info")
def info():
    return jsonify({
        "fast_mode":  FAST_MODE,
        "device":     str(DEVICE),
        "image_size": IMAGE_SIZE,
        "cuda":       torch.cuda.is_available(),
    })

@app.route("/health")
def health():
    """Railway / Docker health-check endpoint.
    Returns 200 only after model warm-up has finished."""
    if _model_ready:
        return jsonify({"status": "ready", "device": str(DEVICE)}), 200
    return jsonify({"status": "loading", "message": "Neural models are loading, please wait…"}), 503

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🎨 Neural Style Transfer Web App")
    print(f"   Mode   : {'⚡ Fast (AdaIN)' if FAST_MODE else '🐢 Optimization'}")
    print(f"   Device : {DEVICE}")
    print(f"   Open   : http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
