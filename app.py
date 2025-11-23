# app.py
import streamlit as st
import torch
import numpy as np
from my_models.vae import VAE

# ============================
# Page Config & Style
# ============================
st.set_page_config(page_title="MNIST VAE Explorer", layout="wide")
st.title("MNIST VAE — 2D Latent Space Explorer")

st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stSlider > div > div > div {background: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

# ============================
# Load Model
# ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "trained_model/vae_mnist.pth"

@st.cache_resource
def load_model():
    model = VAE(latent_dim=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Train the model first.")
    st.stop()

# ============================
# Layout: Left = sliders, Middle = digit, Right = grid
# ============================
col_left, col_mid, col_right = st.columns([1, 1, 3])

# -------- LEFT: Controls --------
with col_left:
    st.markdown("### Latent Space Controls")

    z1 = st.slider("Z₁ (Horizontal)", -3.0, 3.0, 0.0, step=0.05)
    z2 = st.slider("Z₂ (Vertical)", -3.0, 3.0, 0.0, step=0.05)

    st.markdown("---")
    st.info(f"**Current latent vector:**\n`[{z1:.3f}, {z2:.3f}]`")

# -------- MIDDLE: Single Generated Digit --------
with col_mid:
    st.markdown("### Generated Digit")

    z = torch.tensor([[z1, z2]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        generated = model.decode(z).cpu().view(28, 28).numpy()

    st.image(
        generated,
        width=150,
        clamp=True,
        caption=f"Digit for z = [{z1:.2f}, {z2:.2f}]"
    )

# -------- RIGHT: 10×10 Grid (centered around z1, z2) --------
with col_right:
    st.markdown("### Latent Space Grid (Centered on Your z₁, z₂)")

    grid_size = 10
    span = 1.5   # how far the grid spreads around your point

    # Grid shifts dynamically with slider
    z1_grid = z1 + np.linspace(-span, span, grid_size)
    z2_grid = z2 + np.linspace(span, -span, grid_size)   # inverted top→bottom

    cols = st.columns(grid_size)

    with torch.no_grad():
        for i in range(grid_size):
            with cols[i]:
                for j in range(grid_size):
                    z_grid = torch.tensor([[z1_grid[i], z2_grid[j]]],
                                          dtype=torch.float32).to(DEVICE)

                    digit = model.decode(z_grid).cpu().view(28, 28).numpy()

                    st.image(
                        digit,
                        width=40,   # small, crisp
                        clamp=True
                    )

# Footer
st.markdown("---")
st.caption("Variational Autoencoder • 2D latent space • MNIST • Interactive Explorer")
