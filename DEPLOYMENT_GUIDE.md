# 🌐 HOW TO DEPLOY NEURAL STYLE TRANSFER TO THE WEB

This guide provides simple, step-by-step instructions to deploy your Neural Style Transfer web application online so anyone in the world can use it from their browser.

---

## 🏆 Option 1: Hugging Face Spaces (Recommended & 100% Free for AI)

Hugging Face Spaces is the premier free platform for hosting deep learning web applications.
- **Hardware:** 16 GB RAM + 2 vCPU (Free forever)
- **SSL Certificate:** Free automatic HTTPS
- **Custom Link:** `https://huggingface.co/spaces/YOUR_USERNAME/neural-style-transfer`

### Step-by-Step:
1. Create a free account at [huggingface.co](https://huggingface.co) if you haven't already.
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Fill in the details:
   - **Space Name:** `neural-style-transfer`
   - **License:** `mit` or `apache-2.0`
   - **Space SDK:** Select **Docker** (Blank)
   - **Space Hardware:** Select the default **Free CPU Basic (2 vCPU · 16 GB RAM)**
   - **Visibility:** `Public`
4. Click **Create Space**.
5. Push your code to Hugging Face:
   - Copy the Git remote URL shown on your new Space page (e.g. `https://huggingface.co/spaces/YOUR_USERNAME/neural-style-transfer`).
   - In Command Prompt inside your project folder, run:
     ```cmd
     git remote add space https://huggingface.co/spaces/YOUR_USERNAME/neural-style-transfer
     git push space main
     ```
6. Hugging Face will automatically detect your `Dockerfile`, build the container, and launch your live web app in ~2 minutes!

---

## 🚀 Option 2: Render (Render.com)

Render connects directly to your GitHub repository and automatically redeploys whenever you push updates.

### Step-by-Step:
1. Sign up / Log in to [render.com](https://render.com) using your GitHub account.
2. Click **New +** in the dashboard and choose **Web Service**.
3. Under *Connect a repository*, choose **`vaibhavsharma0038-pixel/Neural-Style-Transfer1`**.
4. Configure the settings:
   - **Name:** `neural-style-studio`
   - **Region:** Closest to your users (e.g., Singapore or Frankfurt)
   - **Branch:** `main`
   - **Runtime:** `Python 3` (or choose `Docker`)
   - **Build Command:**
     ```bash
     pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
     ```
   - **Start Command:**
     ```bash
     gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180
     ```
   - **Instance Type:** `Free`
5. Click **Deploy Web Service**.
6. Render will build and host your app with a free custom URL like `https://neural-style-studio.onrender.com`.

---

## 🚂 Option 3: Railway (Railway.app)

Railway is an intuitive cloud platform that auto-detects Docker and Python projects.

### Step-by-Step:
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **+ New Project** → **Deploy from GitHub repo**.
3. Select **`vaibhavsharma0038-pixel/Neural-Style-Transfer1`**.
4. Railway will automatically detect the `Procfile` / `Dockerfile` and start building.
5. In your project settings, click **Generate Domain** to get a public URL (e.g., `neural-style-transfer.up.railway.app`).

---

## 🐳 Option 4: Deploy Anywhere with Docker (AWS / GCP / DigitalOcean)

If you have a Linux virtual machine (Ubuntu VPS, AWS EC2, or DigitalOcean Droplet):

1. **Install Docker:**
   ```bash
   sudo apt update && sudo apt install docker.io -y
   ```
2. **Clone your repository:**
   ```bash
   git clone https://github.com/vaibhavsharma0038-pixel/Neural-Style-Transfer1.git
   cd Neural-Style-Transfer1
   ```
3. **Build the Docker container:**
   ```bash
   docker build -t neural-style-transfer .
   ```
4. **Run the container:**
   ```bash
   docker run -d --restart always -p 80:7860 --name nst-app neural-style-transfer
   ```
5. Your app is now live at `http://YOUR_SERVER_IP`!

---

## 💡 Files Added For Deployment

- **[Dockerfile](file:///c:/Users/Vaibhav%20Sharma/Downloads/Neural-Style-Transfer1/Dockerfile):** CPU-optimized lightweight Docker image configuration.
- **[Procfile](file:///c:/Users/Vaibhav%20Sharma/Downloads/Neural-Style-Transfer1/Procfile):** Production Gunicorn process runner for cloud platforms.
- **[requirements.txt](file:///c:/Users/Vaibhav%20Sharma/Downloads/Neural-Style-Transfer1/requirements.txt):** Includes `gunicorn` for production WSGI serving.
- **[server.py](file:///c:/Users/Vaibhav%20Sharma/Downloads/Neural-Style-Transfer1/server.py):** Dynamically binds to the assigned cloud environment `PORT`.
