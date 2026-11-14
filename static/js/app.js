/* ══════════════════════════════════════════════════════════════
   NeuralStyle Studio — Frontend Controller
   Interactive Split Slider, Masterpiece Inspirations, Webcam & Art Lab
   ══════════════════════════════════════════════════════════════ */

// ── Application State ────────────────────────────────────────────────────────
const state = {
  contentFile: null,
  styleFile: null,
  jobId: null,
  pollTimer: null,
  resolution: 256,
  preserveColor: false,
  webcamStream: null,
  sliderPct: 50,
  isDraggingSlider: false,
  filters: {
    brightness: 100,
    contrast: 100,
    saturate: 100,
    warmth: 0,
  },
};

// ── DOM Helper ───────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── Application Initialization ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupDragAndDrop("contentDrop", "content");
  setupDragAndDrop("styleDrop", "style");
  setupFileInput("contentFile", "content");
  setupFileInput("styleFile", "style");
  setupSliders();
  setupResolutionToggles();
  setupColorPreservation();
  setupStyleInspirations();
  setupWebcamModal();
  setupSplitSlider();
  setupViewModeTabs();
  setupArtLabFilters();
  setupExportTools();
  setupSmoothScrolling();
  updateGenerateButtonState();
});

// ── Interactive Sliders ─────────────────────────────────────────────────────
function setupSliders() {
  const alphaSlider = $("alpha");
  if (alphaSlider) {
    alphaSlider.addEventListener("input", () => {
      const pct = Math.round(alphaSlider.value * 100);
      $("alphaVal").textContent = `${pct}%`;
    });
  }
}

// ── Resolution / Quality Toggle ─────────────────────────────────────────────
function setupResolutionToggles() {
  const container = $("resolutionGroup");
  if (!container) return;

  const buttons = container.querySelectorAll(".chip-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.resolution = parseInt(btn.getAttribute("data-res"), 10) || 256;
    });
  });
}

// ── Natural Color Preservation ──────────────────────────────────────────────
function setupColorPreservation() {
  const toggle = $("preserveColorToggle");
  const label = $("colorToggleLabel");
  if (!toggle || !label) return;

  toggle.addEventListener("change", () => {
    state.preserveColor = toggle.checked;
    label.textContent = toggle.checked ? "Natural Palette: ON" : "Natural Palette: OFF";
  });
}

// ── Quick Masterpiece Inspirations ──────────────────────────────────────────
function setupStyleInspirations() {
  const pills = document.querySelectorAll(".art-pill");
  pills.forEach((pill) => {
    pill.addEventListener("click", async () => {
      const filename = pill.getAttribute("data-file");
      if (!filename) return;

      // Mark pill active
      pills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");

      try {
        const res = await fetch(`/preset/style/${filename}`);
        if (!res.ok) throw new Error("Could not load preset style");

        const blob = await res.blob();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
        state.styleFile = file;

        // Display in dropzone
        const dropzone = $("styleDrop");
        const previewWrap = $("stylePreviewWrap");
        const previewImg = $("stylePreview");
        const emptyView = dropzone.querySelector(".dropzone-empty");

        previewImg.src = URL.createObjectURL(blob);
        emptyView.classList.add("hidden");
        previewWrap.classList.remove("hidden");
        dropzone.classList.add("has-media");

        updateGenerateButtonState();
      } catch (err) {
        console.error("Error loading inspiration style:", err);
      }
    });
  });
}

// ── Webcam Selfie Capture ───────────────────────────────────────────────────
function setupWebcamModal() {
  const openBtn = $("openWebcamBtn");
  const modal = $("webcamModal");
  const closeBtn = $("closeWebcamBtn");
  const cancelBtn = $("cancelWebcamBtn");
  const captureBtn = $("captureWebcamBtn");
  const video = $("webcamVideo");
  const canvas = $("webcamCanvas");

  if (!openBtn || !modal) return;

  const stopCamera = () => {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((track) => track.stop());
      state.webcamStream = null;
    }
    video.srcObject = null;
    modal.classList.add("hidden");
  };

  openBtn.addEventListener("click", async () => {
    try {
      modal.classList.remove("hidden");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 720 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      state.webcamStream = stream;
      video.srcObject = stream;
      video.play();
    } catch (err) {
      alert("Unable to access camera: " + err.message);
      modal.classList.add("hidden");
    }
  });

  closeBtn.addEventListener("click", stopCamera);
  cancelBtn.addEventListener("click", stopCamera);

  captureBtn.addEventListener("click", () => {
    if (!video.videoWidth || !video.videoHeight) return;

    // Draw square crop from center of video frame
    const size = Math.min(video.videoWidth, video.videoHeight);
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");

    // Mirror horizontal to match preview
    ctx.translate(size, 0);
    ctx.scale(-1, 1);

    const sx = (video.videoWidth - size) / 2;
    const sy = (video.videoHeight - size) / 2;
    ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);

    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], "selfie_capture.png", { type: "image/png" });
      handleImageSelection("content", file);
      stopCamera();
    }, "image/png");
  });
}

// ── Drag & Drop Handlers ────────────────────────────────────────────────────
function setupDragAndDrop(dropzoneId, type) {
  const dropzone = $(dropzoneId);
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleImageSelection(type, file);
    }
  });
}

// ── File Input Handlers ─────────────────────────────────────────────────────
function setupFileInput(inputId, type) {
  const input = $(inputId);
  if (!input) return;

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) {
      handleImageSelection(type, input.files[0]);
    }
  });
}

// ── Process Selected Image ──────────────────────────────────────────────────
function handleImageSelection(type, file) {
  state[`${type}File`] = file;

  const dropzone = $(`${type}Drop`);
  const previewWrap = $(`${type}PreviewWrap`);
  const previewImg = $(`${type}Preview`);
  const emptyView = dropzone.querySelector(".dropzone-empty");

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    emptyView.classList.add("hidden");
    previewWrap.classList.remove("hidden");
    dropzone.classList.add("has-media");
  };
  reader.readAsDataURL(file);

  updateGenerateButtonState();
}

// ── Clear Image ─────────────────────────────────────────────────────────────
function clearImage(type) {
  state[`${type}File`] = null;

  const dropzone = $(`${type}Drop`);
  const previewWrap = $(`${type}PreviewWrap`);
  const previewImg = $(`${type}Preview`);
  const emptyView = dropzone.querySelector(".dropzone-empty");
  const fileInput = $(`${type}File`);

  previewImg.src = "";
  previewWrap.classList.add("hidden");
  emptyView.classList.remove("hidden");
  dropzone.classList.remove("has-media");

  if (fileInput) fileInput.value = "";

  // Also remove active status from inspiration pills if clearing style
  if (type === "style") {
    document.querySelectorAll(".art-pill").forEach((p) => p.classList.remove("active"));
  }

  updateGenerateButtonState();
}

// ── Synthesis Button Availability ───────────────────────────────────────────
function updateGenerateButtonState() {
  const btn = $("generateBtn");
  if (!btn) return;
  const isReady = !!state.contentFile && !!state.styleFile;
  btn.disabled = !isReady;
}

// ── Execute Synthesis Request ───────────────────────────────────────────────
async function startTransfer() {
  const btn = $("generateBtn");
  if (btn.disabled) return;

  btn.disabled = true;
  $("btnText").textContent = "Submitting to GPU/CPU...";
  $("btnIcon").textContent = "⏳";

  // Hide any previous result & show progress bar
  $("resultWrap").classList.add("hidden");
  $("progressWrap").classList.remove("hidden");
  resetMilestones();
  updateProgress(5, "Streaming images to neural pipeline...");

  // Build multipart form
  const formData = new FormData();
  formData.append("content_file", state.contentFile);
  formData.append("style_file", state.styleFile);
  formData.append("alpha", $("alpha") ? $("alpha").value : 1.0);
  formData.append("size", state.resolution || 256);
  formData.append("preserve_color", state.preserveColor ? "true" : "false");
  if ($("steps")) formData.append("steps", $("steps").value);

  try {
    const res = await fetch("/stylize", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    state.jobId = data.job_id;
    $("btnText").textContent = "Synthesizing Artwork...";
    pollJobStatus(state.jobId);
  } catch (err) {
    updateProgress(0, `Error: ${err.message}`);
    btn.disabled = false;
    $("btnText").textContent = "Try Again";
    $("btnIcon").textContent = "⚠️";
  }
}

// ── Poll Server Status ───────────────────────────────────────────────────────
function pollJobStatus(jobId) {
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/status/${jobId}`);
      const data = await res.json();

      updateProgress(data.progress || 10, data.message || "Processing neural layers...");
      updateMilestones(data.progress || 10);

      if (data.status === "done") {
        clearInterval(state.pollTimer);
        displayFinalResult(jobId, data.elapsed);
      } else if (data.status === "error") {
        clearInterval(state.pollTimer);
        updateProgress(0, `❌ ${data.message || "Synthesis error"}`);
        $("generateBtn").disabled = false;
        $("btnText").textContent = "Try Again";
        $("btnIcon").textContent = "⚠️";
      }
    } catch (e) {
      // transient network tick, keep polling
    }
  }, 400);
}

// ── Display Completed Result ────────────────────────────────────────────────
function displayFinalResult(jobId, elapsed) {
  const resultUrl = `/result/${jobId}?t=${Date.now()}`;
  let contentUrl = "";
  let styleUrl = "";

  if (state.contentFile) {
    contentUrl = URL.createObjectURL(state.contentFile);
    $("resultContent").src = contentUrl;
    $("sliderContentImg").src = contentUrl;
  }
  if (state.styleFile) {
    styleUrl = URL.createObjectURL(state.styleFile);
    $("resultStyle").src = styleUrl;
  }

  // Load synthesized outputs
  $("resultOutput").src = resultUrl;
  $("sliderResultImg").src = resultUrl;
  $("downloadBtn").href = `/download/${jobId}`;
  
  if ($("resultTime")) {
    $("resultTime").textContent = elapsed ? `⚡ Rendered in ${elapsed}s` : "⚡ Instant Render";
  }

  // Sync split slider width
  syncSliderDimensions();

  // Reveal showcase
  $("progressWrap").classList.add("hidden");
  $("resultWrap").classList.remove("hidden");
  $("resultWrap").scrollIntoView({ behavior: "smooth", block: "start" });

  // Reset synthesis button
  const btn = $("generateBtn");
  btn.disabled = false;
  $("btnText").textContent = "Synthesize Stylized Artwork";
  $("btnIcon").textContent = "⚡";
}

// ── Split Slider Controller ─────────────────────────────────────────────────
function setupSplitSlider() {
  const slider = $("compSlider");
  const overLayer = $("sliderOverLayer");
  const divider = $("sliderDivider");
  if (!slider || !overLayer || !divider) return;

  const updatePosition = (clientX) => {
    const rect = slider.getBoundingClientRect();
    const x = clientX - rect.left;
    let pct = (x / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    state.sliderPct = pct;

    overLayer.style.width = `${pct}%`;
    divider.style.left = `${pct}%`;
  };

  const onPointerDown = (e) => {
    state.isDraggingSlider = true;
    updatePosition(e.clientX || (e.touches && e.touches[0].clientX));
  };

  const onPointerMove = (e) => {
    if (!state.isDraggingSlider) return;
    updatePosition(e.clientX || (e.touches && e.touches[0].clientX));
  };

  const onPointerUp = () => {
    state.isDraggingSlider = false;
  };

  slider.addEventListener("mousedown", onPointerDown);
  slider.addEventListener("touchstart", onPointerDown, { passive: true });

  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("touchmove", onPointerMove, { passive: true });

  window.addEventListener("mouseup", onPointerUp);
  window.addEventListener("touchend", onPointerUp);

  window.addEventListener("resize", syncSliderDimensions);
}

function syncSliderDimensions() {
  const slider = $("compSlider");
  const overImg = $("sliderResultImg");
  if (!slider || !overImg) return;
  const width = slider.offsetWidth;
  overImg.style.width = `${width}px`;
}

// ── View Mode Switcher (Slider vs Grid) ──────────────────────────────────────
function setupViewModeTabs() {
  const tabs = document.querySelectorAll(".view-tab");
  const sliderView = $("sliderViewContainer");
  const gridView = $("gridViewContainer");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const mode = tab.getAttribute("data-mode");
      if (mode === "slider") {
        sliderView.classList.remove("hidden");
        gridView.classList.add("hidden");
        syncSliderDimensions();
      } else {
        sliderView.classList.add("hidden");
        gridView.classList.remove("hidden");
      }
    });
  });
}

// ── Interactive Art Lab (Post-Processing Filters) ───────────────────────────
function setupArtLabFilters() {
  const brightness = $("filterBrightness");
  const contrast = $("filterContrast");
  const saturate = $("filterSaturate");
  const warmth = $("filterWarmth");
  const resetBtn = $("resetFiltersBtn");

  const applyFilters = () => {
    state.filters.brightness = brightness ? brightness.value : 100;
    state.filters.contrast = contrast ? contrast.value : 100;
    state.filters.saturate = saturate ? saturate.value : 100;
    state.filters.warmth = warmth ? warmth.value : 0;

    $("brightnessVal").textContent = `${state.filters.brightness}%`;
    $("contrastVal").textContent = `${state.filters.contrast}%`;
    $("saturateVal").textContent = `${state.filters.saturate}%`;
    $("warmthVal").textContent = `${state.filters.warmth}%`;

    const filterStr = `brightness(${state.filters.brightness}%) contrast(${state.filters.contrast}%) saturate(${state.filters.saturate}%) sepia(${state.filters.warmth}%)`;

    if ($("sliderResultImg")) $("sliderResultImg").style.filter = filterStr;
    if ($("resultOutput")) $("resultOutput").style.filter = filterStr;
  };

  [brightness, contrast, saturate, warmth].forEach((el) => {
    if (el) el.addEventListener("input", applyFilters);
  });

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (brightness) brightness.value = 100;
      if (contrast) contrast.value = 100;
      if (saturate) saturate.value = 100;
      if (warmth) warmth.value = 0;
      applyFilters();
    });
  }
}

// ── Export Tools (Download with Filters, Copy Clipboard, Lightbox) ──────────
function setupExportTools() {
  // 1-Click Copy Image to Clipboard
  const copyBtn = $("copyClipboardBtn");
  const copyText = $("copyText");

  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const img = $("resultOutput");
      if (!img || !img.src) return;

      try {
        const response = await fetch(img.src);
        const blob = await response.blob();
        await navigator.clipboard.write([
          new ClipboardItem({ [blob.type || "image/png"]: blob }),
        ]);

        copyText.textContent = "✓ Copied to Clipboard!";
        copyBtn.style.borderColor = "var(--accent-emerald)";
        setTimeout(() => {
          copyText.textContent = "Copy Image";
          copyBtn.style.borderColor = "";
        }, 2000);
      } catch (err) {
        console.error("Clipboard write error:", err);
        alert("Clipboard copy not supported or permission denied. Please use Download PNG.");
      }
    });
  }

  // Fullscreen Lightbox
  const lightboxModal = $("lightboxModal");
  const openLightboxBtn = $("openLightboxBtn");
  const closeLightboxBtn = $("closeLightboxBtn");
  const lightboxImg = $("lightboxImg");

  const openLightbox = () => {
    const src = $("resultOutput") ? $("resultOutput").src : null;
    if (src && lightboxImg && lightboxModal) {
      lightboxImg.src = src;
      lightboxModal.classList.remove("hidden");
    }
  };

  if (openLightboxBtn) openLightboxBtn.addEventListener("click", openLightbox);
  if (closeLightboxBtn) closeLightboxBtn.addEventListener("click", () => lightboxModal.classList.add("hidden"));
  if (lightboxModal) {
    lightboxModal.addEventListener("click", (e) => {
      if (e.target === lightboxModal) lightboxModal.classList.add("hidden");
    });
  }

  // Canvas-rendered Download with active Art Lab filters
  const downloadBtn = $("downloadBtn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", (e) => {
      const f = state.filters;
      // If user modified filters, render to canvas before saving
      if (f.brightness !== 100 || f.contrast !== 100 || f.saturate !== 100 || f.warmth !== 0) {
        e.preventDefault();
        const img = $("resultOutput");
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth || 512;
        canvas.height = img.naturalHeight || 512;
        const ctx = canvas.getContext("2d");
        ctx.filter = `brightness(${f.brightness}%) contrast(${f.contrast}%) saturate(${f.saturate}%) sepia(${f.warmth}%)`;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        const a = document.createElement("a");
        a.download = `neural_art_${state.jobId || "custom"}.png`;
        a.href = canvas.toDataURL("image/png");
        a.click();
      }
    });
  }
}

// ── Reset Studio Workspace ──────────────────────────────────────────────────
function resetForm() {
  clearImage("content");
  clearImage("style");
  $("resultWrap").classList.add("hidden");
  $("progressWrap").classList.add("hidden");
  updateGenerateButtonState();

  const studio = document.getElementById("studio");
  if (studio) studio.scrollIntoView({ behavior: "smooth" });
}

// ── Progress Meter & Milestones ──────────────────────────────────────────────
function updateProgress(percentage, message) {
  $("progressFill").style.width = `${percentage}%`;
  $("progressPct").textContent = `${percentage}%`;
  $("progressStatus").textContent = message;
}

function resetMilestones() {
  ["step1", "step2", "step3", "step4"].forEach((id) => {
    const el = $(id);
    if (el) el.classList.remove("active", "done");
  });
}

function updateMilestones(pct) {
  const steps = [
    { id: "step1", from: 0, to: 25 },
    { id: "step2", from: 25, to: 65 },
    { id: "step3", from: 65, to: 90 },
    { id: "step4", from: 90, to: 100 },
  ];

  steps.forEach((s) => {
    const el = $(s.id);
    if (!el) return;
    if (pct >= s.to) {
      el.classList.add("done");
      el.classList.remove("active");
    } else if (pct >= s.from) {
      el.classList.add("active");
      el.classList.remove("done");
    } else {
      el.classList.remove("active", "done");
    }
  });
}

// ── Smooth Anchor Scrolling ─────────────────────────────────────────────────
function setupSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const targetId = anchor.getAttribute("href");
      if (targetId === "#") return;
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();
        targetElement.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
}
