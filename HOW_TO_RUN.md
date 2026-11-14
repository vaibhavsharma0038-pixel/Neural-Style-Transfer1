# HOW TO RUN THE PROJECT

This guide explains how to start, use, and stop the Neural Style Transfer web application on Windows using **Command Prompt (CMD)**, **PowerShell**, or the **One-Click launcher**.

---

## ⚡ Method 1: One-Click Run (Easiest)

In the project folder, simply **double-click** the file:
```cmd
run_app.bat
```

**What this does automatically:**
1. Verifies that Python is installed.
2. Opens your default web browser to `http://localhost:5000`.
3. Launches the Flask backend server in its own dedicated terminal window.

---

## 💻 Method 2: Run via Command Prompt (CMD)

If you prefer to run it manually inside **CMD**:

### Step 1: Open Command Prompt
- Press <kbd>Win</kbd> + <kbd>R</kbd> on your keyboard.
- Type `cmd` and press **Enter**.

### Step 2: Navigate to the Project Folder
Copy and paste the command below into CMD and press **Enter**:
```cmd
cd "C:\Users\Vaibhav Sharma\Downloads\Neural-Style-Transfer1"
```

### Step 3: *(First Time Only)* Install Dependencies
If you haven't installed the required Python packages yet, run:
```cmd
pip install -r requirements.txt
```

### Step 4: Start the Server
You have two options:

- **Direct Python command:**
  ```cmd
  python server.py
  ```
- **Or trigger the batch runner:**
  ```cmd
  run_app.bat
  ```

### Step 5: Open in Your Browser
Visit the application in Chrome, Edge, Brave, or any browser:
```
http://localhost:5000
```
*(or `http://127.0.0.1:5000`)*

---

## 🟦 Method 3: Run via PowerShell

1. Open **PowerShell**.
2. Navigate to the project directory:
   ```powershell
   cd "C:\Users\Vaibhav Sharma\Downloads\Neural-Style-Transfer1"
   ```
3. Start the server:
   ```powershell
   python server.py
   ```
4. Open your browser and navigate to `http://localhost:5000`.

---

## 🛑 How to Stop the App

To stop the web server at any time:
1. Click into the Command Prompt or PowerShell window where the server is running.
2. Press:
   ```cmd
   Ctrl + C
   ```
3. Alternatively, you can simply close the terminal window.

---

## 🧠 Model Information & Features

- **Pretrained Checkpoint:** `adain_decoder.pth` (already fine-tuned on high-resolution WikiArt master paintings).
- **Inference Speed:** ~1.5–3 seconds per image on CPU.
- **Interactive UI:**
  - ↔ Draggable Before/After Split Comparison Slider.
  - ⚡ 1-Click Masterpiece Inspirations (Van Gogh, Klimt, Monet, Munch, Kandinsky, Dalí, Pollock, Picasso).
  - 📸 Live Webcam Selfie Capture.
  - 🎨 Natural Color Preservation toggle.
  - ✨ Art Lab post-processing fine-tuning sliders.
  - 📋 1-Click Copy Image to Clipboard & Fullscreen Lightbox.

---

## 🔄 How to Retrain the Model (Optional)

If you ever want to fine-tune or train on additional artwork images:
```cmd
python train_fast.py
```
This runs 15 high-resolution epochs and updates `adain_decoder.pth`.
