import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

from model import vgg, Decoder, device
from utils import adain, calc_mean_std

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

# Load images
def load_images(folder):
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("RGB")
            images.append(transform(img))
    return images

content_images = load_images("data/content")
style_images = load_images("data/style")

print("Content Images:", len(content_images))
print("Style Images:", len(style_images))

# Model
decoder = Decoder().to(device)
optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

EPOCHS = 20

# Training
for epoch in range(EPOCHS):
    total_loss = 0

    for content_img in content_images:
        style_img = random.choice(style_images)

        content = content_img.unsqueeze(0).to(device)
        style = style_img.unsqueeze(0).to(device)

        content_feat = vgg(content)
        style_feat = vgg(style)

        t = adain(content_feat, style_feat)
        output = decoder(t)

        output_feat = vgg(output)

        # Content loss
        content_loss = torch.mean((output_feat - t) ** 2)

        # Style loss
        out_mean, out_std = calc_mean_std(output_feat)
        style_mean, style_std = calc_mean_std(style_feat)

        style_loss = torch.mean(
            (out_mean - style_mean) ** 2 +
            (out_std - style_std) ** 2
        )

        # Identity loss
        identity = decoder(vgg(content))
        identity_loss = torch.mean((identity - content) ** 2)

        # Total loss
        loss = content_loss + 10 * style_loss + 5 * identity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.2f}")

# Save model
torch.save(decoder.state_dict(), "model.pth")
print("Model Saved!")

# Test output
content = content_images[0].unsqueeze(0).to(device)
style = style_images[0].unsqueeze(0).to(device)

with torch.no_grad():
    t = adain(vgg(content), vgg(style))
    output = decoder(t)

img = output.squeeze().permute(1, 2, 0).cpu().numpy()

plt.imshow(img)
plt.title("Stylized Output")
plt.axis("off")
plt.show()
