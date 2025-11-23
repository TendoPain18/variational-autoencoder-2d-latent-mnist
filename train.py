# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from my_models.vae import VAE
from utils.loss import vae_loss
from data.loader import get_mnist_loaders
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "trained_model/vae_mnist.pth"

print(f"Using device: {DEVICE}")
print("Loading MNIST data...")
train_loader, _ = get_mnist_loaders(BATCH_SIZE)

model = VAE(latent_dim=2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE).view(-1, 784)

        recon, mu, logvar = model(data)
        loss = vae_loss(recon, data, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Save model
os.makedirs("trained_model", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")