# 🎨 Neural Style Transfer

A deep learning-based **Neural Style Transfer (NST)** project that combines the content of one image with the artistic style of another image to generate a new stylized image.

The project uses a pre-trained **VGG-19** convolutional neural network to extract content and style features. Style information is represented using **Gram Matrices**, and the generated image is optimized using **LBFGS** optimization.

---

## 📌 Project Overview

Neural Style Transfer is a computer vision technique that generates artistic images by combining:

* 🖼️ **Content Image** – Provides the structure and objects of the image.
* 🎨 **Style Image** – Provides artistic characteristics such as colors, textures, and patterns.
* ✨ **Generated Image** – Combines the content and style information.

The project uses a pre-trained VGG-19 network to extract feature representations from the images.

---

## 🚀 Features

* Neural Style Transfer using **VGG-19**
* Content and style feature extraction
* **Gram Matrix** for style representation
* **LBFGS** optimization
* Multiple artistic style presets
* Custom content and style image support
* Image preprocessing and post-processing
* Generated image visualization
* Content loss and style loss calculation
* **PSNR** evaluation
* **SSIM** evaluation
* Output image saving
* Training/optimization metrics logging

---

## 🎨 Available Style Presets

The project supports multiple artistic styles:

| Style            | Description                         |
| ---------------- | ----------------------------------- |
| 🖌️ Oil Paint    | Oil painting-inspired appearance    |
| 💧 Watercolor    | Soft watercolor effect              |
| 🌻 Van Gogh      | Inspired by Van Gogh-style textures |
| ✏️ Pencil Sketch | Pencil drawing effect               |
| 🖤 Charcoal      | Charcoal-style artistic effect      |
| 🎨 Impressionist | Impressionist painting effect       |
| 🧩 Mosaic        | Mosaic-like artistic texture        |
| 🌈 Neon Glow     | Bright neon-style appearance        |
| 🖼️ Custom       | Use your own style image            |

---

## 🧠 Model Architecture

The project uses **VGG-19**, a pre-trained convolutional neural network.

Selected layers are used to extract content and style features.

### Content Layer

```text
conv_4
```

### Style Layers

```text
conv_1
conv_2
conv_3
conv_4
conv_5
```

The content layer captures the high-level structure of the image, while the style layers capture textures, colors, patterns, and artistic features.

---

## 🔢 Gram Matrix

The **Gram Matrix** is used to represent the style of an image.

For a feature map `F`, the Gram Matrix can be represented as:

```text
G = F × Fᵀ
```

It captures correlations between feature maps and helps represent the visual style independently of the exact spatial arrangement of objects.

---

## ⚙️ Optimization

The generated image is optimized by minimizing a combined loss function:

```text
Total Loss = α × Content Loss + β × Style Loss
```

Where:

* **Content Loss** measures the difference between content features.
* **Style Loss** measures the difference between style representations.
* **α** controls the importance of content.
* **β** controls the importance of style.

The project uses **LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)** optimization for image optimization.

---

## 📊 Evaluation Metrics

The project evaluates generated images using:

### PSNR

**Peak Signal-to-Noise Ratio (PSNR)** measures the similarity between two images based on pixel-level differences.

Higher PSNR generally indicates lower pixel-level error.

### SSIM

**Structural Similarity Index (SSIM)** measures structural similarity between images.

SSIM considers factors such as:

* Luminance
* Contrast
* Structure

The value generally ranges from:

```text
0 → Low similarity
1 → High similarity
```

> Note: Accuracy is not an appropriate primary metric for Neural Style Transfer because NST is a generative image transformation task rather than a classification task.

---

## 📁 Project Structure

```text
NST_Project/
│
├── main.py
├── model.py
├── utils.py
├── requirements.txt
│
├── data/
│   ├── content/
│   │   └── content images
│   │
│   └── style/
│       └── style images
│
├── output/
│   └── generated images
│
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* OpenCV
* Pillow
* Matplotlib
* Scikit-Image

### Deep Learning

* VGG-19
* Convolutional Neural Networks
* Gram Matrix
* Feature Extraction
* Image Optimization

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NST_Project.git
```

Navigate into the project:

```bash
cd NST_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the main Python file:

```bash
python main.py
```

Provide/select:

1. Content image
2. Style image or style preset
3. Output location
4. Optimization parameters

The generated image will be saved in the `output/` directory.

---

## 🖼️ Input and Output

### Input

The model requires:

```text
Content Image + Style Image
```

### Output

The system generates:

```text
Stylized Image
```

For example:

```text
Content Image
      +
Style Image
      ↓
Neural Style Transfer
      ↓
Generated Artistic Image
```

---

## 📈 Loss Functions

The project calculates two major losses.

### Content Loss

Content loss ensures that the generated image maintains the structural information of the content image.

```text
Content Loss = ||Generated Features - Content Features||²
```

### Style Loss

Style loss compares Gram Matrices of the generated and style images.

```text
Style Loss = Σ ||G_generated - G_style||²
```

### Total Loss

```text
Total Loss = Content Weight × Content Loss
           + Style Weight × Style Loss
```

---

## 🎯 Applications

Neural Style Transfer can be used in:

* 🎨 Digital art generation
* 📸 Photo stylization
* 🖼️ Artistic image transformation
* 🎬 Visual effects
* 🎮 Game graphics
* ✨ Creative AI applications
* 📱 Image editing applications

---

## 🔮 Future Improvements

Possible improvements include:

* Real-time style transfer
* Faster feed-forward style transfer
* SANet-based style transfer
* AdaIN-based style transfer
* Multi-style image generation
* Web-based interface
* Streamlit/Gradio application
* GPU acceleration
* Improved perceptual quality
* Automated hyperparameter tuning

---

## 📊 Results

The model generates artistic images by preserving the structure of the content image while transferring visual characteristics from the selected style image.

Example workflow:

```text
              ┌─────────────────┐
              │  Content Image  │
              └────────┬────────┘
                       │
                       ▼
                 VGG-19 Features
                       │
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Content Features    Style Features
              │                 │
              │          Gram Matrix
              │                 │
              └────────┬────────┘
                       ▼
                  Loss Function
                       │
                       ▼
                    LBFGS
                       │
                       ▼
              Generated Image
```

---

## 👨‍💻 Author

**Vaibhav Sharma**

B.Tech Computer Science / Artificial Intelligence & Machine Learning

---

## 📜 License

This project is intended for educational and research purposes.

---

## ⭐ Acknowledgements

This project is based on the concepts introduced in the original Neural Style Transfer research:

**"A Neural Algorithm of Artistic Style"**

by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
