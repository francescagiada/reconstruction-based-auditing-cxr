"""ae_eval_script.py

Evaluate a pretrained autoencoder on a list of images and compute per-image metrics.

Usage (PowerShell):
python ae_eval_script.py --checkpoint path/to/model.pth --csv images.csv --output results.csv --base-dir "C:/path/to/images"

The script implements these modular functions:
  - load_model()
  - preprocess_image()
  - reconstruct_image()
  - compute_mse(), compute_ssim()
  - compute_ppw()
  - compute_edge_difference_normalized()
  - main()

Notes:
  - Images are normalized to [-1, 1] consistent with training preprocessing.
  - SSIM uses data_range=2.0 because inputs are in [-1, 1].
"""

from pathlib import Path
import argparse
import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import scipy.ndimage as ndi
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# Models (same structure as notebook)
class TinyAE(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32,64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128,256,4,2,1),  nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256*14*14, latent)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 256*14*14),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256,14,14)),
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64, 4,2,1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4,2,1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1,  4,2,1), nn.Tanh()
        )
    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out, z


class VariationalAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 28 * 28, latent_dim)
        self.fc_logvar = nn.Linear(128 * 28 * 28, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 128 * 28 * 28)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128, 28, 28)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh(),
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(self.dec_fc(z))
        return x_hat, (mu, logvar, z)


# Utilities / Preprocessing
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path: str, model_type: str = 'tiny', device=None):
    """Load a pretrained model. model_type: 'tiny' or 'vae'."""
    device = device or get_device()
    if model_type.lower().startswith('v'):
        model = VariationalAE(latent_dim=128)
    else:
        model = TinyAE(latent=128)
    model = model.to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(sd)
    except Exception:
        # assume the checkpoint stores a dict with 'model_state' or similar
        if isinstance(sd, dict):
            possible = None
            for key in ('model_state', 'state_dict', 'model_state_dict'):
                if key in sd:
                    possible = sd[key]
                    break
            if possible is not None:
                model.load_state_dict(possible)
            else:
                # last resort: try partial load
                model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
        else:
            raise
    model.eval()
    return model


def preprocess_image(image_path: str, base_dir: str = None, img_size: int = 224):
    """Load a single image and apply the same transforms as the notebook dataset.
    Returns a torch tensor shaped (1, 1, H, W) normalized to [-1, 1]."""
    if base_dir:
        p = Path(base_dir) / image_path
    else:
        p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = Image.open(p).convert('L')
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    t = tf(img).unsqueeze(0)  # (1,1,H,W)
    return t


def tensor_to_numpy_normalized(tensor: torch.Tensor):
    """Convert tensor in [-1,1] to numpy float32 array in [-1,1].
    Accepts shape (1,1,H,W) or (1,H,W) or (1,1,H,W)"""
    with torch.no_grad():
        t = tensor.detach().cpu()
        if t.ndim == 4:
            t = t[0, 0]
        elif t.ndim == 3:
            # (1,H,W)
            t = t[0]
        arr = t.numpy().astype(np.float32)
    # arr already in [-1,1] because of Normalize([0.5],[0.5])
    return arr


class CalibrationDataset(Dataset):
    """Simple dataset that yields preprocessed tensors for calibration.
    Takes an iterable of image paths (relative or absolute) and applies the same
    preprocessing used by `preprocess_image`.
    """
    def __init__(self, paths, base_dir=None, img_size=224):
        self.paths = list(paths)
        self.base_dir = base_dir
        self.img_size = img_size
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        t = preprocess_image(p, base_dir=self.base_dir, img_size=self.img_size)
        return t


def reconstruct_image(model: nn.Module, input_tensor: torch.Tensor, device=None):
    device = device or get_device()
    model = model.to(device)
    x = input_tensor.to(device)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            xhat = out[0]
        else:
            xhat = out
    return xhat.cpu()


# Metrics
def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # skimage expects HxW arrays; data_range for [-1,1] is 2.0
    try:
        return float(ssim(a, b, data_range=2.0))
    except Exception:
        return float('nan')


def compute_ppw(a: np.ndarray, b: np.ndarray, epsilon: float = 0.05) -> float:
    return float(np.mean(np.abs(a - b) < epsilon))


def estimate_epsilon(autoencoder: nn.Module, dataloader: DataLoader, percentile: float = 90, device=None, max_images: int = 500):
    """Estimate epsilon for PPW by computing pixel-wise absolute errors on a calibration set.

    - Runs the autoencoder on up to `max_images` from the dataloader.
    - Collects pixel errors |x - recon| into a single numpy array and returns the specified percentile.
    """
    device = device or get_device()
    autoencoder = autoencoder.to(device)
    errors = []
    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            # batch may be a tensor or a tuple (tensor, ...)
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            out = autoencoder(x)
            xhat = out[0] if isinstance(out, tuple) else out
            # convert to numpy normalized arrays
            x_np = x.detach().cpu().numpy()
            xhat_np = xhat.detach().cpu().numpy()
            B = x_np.shape[0]
            for i in range(B):
                a = tensor_to_numpy_normalized(torch.from_numpy(x_np[i:i+1]))
                b = tensor_to_numpy_normalized(torch.from_numpy(xhat_np[i:i+1]))
                # ensure 2D
                if a.ndim == 3:
                    a = a.squeeze()
                if b.ndim == 3:
                    b = b.squeeze()
                err = np.abs(a - b).ravel()
                errors.append(err)
                processed += 1
                if processed >= max_images:
                    break
            if processed >= max_images:
                break
    if not errors:
        raise RuntimeError('No images processed for epsilon estimation')
    all_errors = np.concatenate(errors, axis=0)
    eps = float(np.percentile(all_errors, percentile))
    return eps


def compute_edge_difference_normalized(a: np.ndarray, b: np.ndarray) -> float:
    # a,b are single-channel 2D arrays in [-1,1]
    # compute gradient magnitude using Sobel
    ax = ndi.sobel(a, axis=0)
    ay = ndi.sobel(a, axis=1)
    bx = ndi.sobel(b, axis=0)
    by = ndi.sobel(b, axis=1)
    Ma = np.hypot(ax, ay)
    Mb = np.hypot(bx, by)
    diff = float(np.mean(np.abs(Ma - Mb)))
    norm = float(np.mean(Ma)) + 1e-8
    return abs(diff / norm)


# CLI / Main loop
def main():
    parser = argparse.ArgumentParser(description='Evaluate AE checkpoint on a CSV of images')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--model-type', choices=['tiny', 'vae'], default='tiny', help='Model architecture type')
    parser.add_argument('--csv', required=True, help='CSV file listing images to evaluate (column default: filename)')
    parser.add_argument('--img-col', default='filename', help='CSV column name that contains image paths')
    parser.add_argument('--base-dir', default=None, help='Optional base directory to prefix image paths')
    parser.add_argument('--output', default='ae_eval_results.csv', help='Output CSV path')
    parser.add_argument('--img-size', type=int, default=224, help='Image size (square) used for preprocessing')
    parser.add_argument('--epsilon', type=float, default=0.05, help='epsilon for PPW metric')
    parser.add_argument('--estimate-epsilon', action='store_true', help='Estimate epsilon automatically from a calibration set')
    parser.add_argument('--calibration-csv', default=None, help='Optional CSV to use for epsilon calibration (if not provided the evaluation CSV will be sampled)')
    parser.add_argument('--calib-size', type=int, default=200, help='Number of images to use for calibration (100-500 recommended)')
    parser.add_argument('--calib-batch-size', type=int, default=16, help='Batch size to use during calibration')
    args = parser.parse_args()

    device = get_device()
    print('Using device:', device)

    # load model
    print('Loading model...')
    model = load_model(args.checkpoint, model_type=args.model_type, device=device)

    # read csv
    df = pd.read_csv(args.csv)
    if args.img_col not in df.columns:
        raise ValueError(f"CSV does not contain column '{args.img_col}'")

    # optionally estimate epsilon using a calibration loader
    epsilon_used = args.epsilon
    if args.estimate_epsilon:
        # build calibration list of image paths
        if args.calibration_csv:
            calib_df = pd.read_csv(args.calibration_csv)
            if args.img_col not in calib_df.columns:
                raise ValueError(f"Calibration CSV does not contain column '{args.img_col}'")
            paths = calib_df[args.img_col].tolist()
        else:
            # reuse evaluation CSV but sample up to calib-size
            sample_df = df.sample(n=min(args.calib_size, max(1, len(df))), random_state=42).reset_index(drop=True)
            paths = sample_df[args.img_col].tolist()

        calib_ds = CalibrationDataset(paths, base_dir=args.base_dir, img_size=args.img_size)
        calib_loader = DataLoader(calib_ds, batch_size=args.calib_batch_size, shuffle=False, num_workers=0)
        print(f'Estimating epsilon using {min(len(paths), args.calib_size)} images (percentile=90)...')
        try:
            epsilon_est = estimate_epsilon(model, calib_loader, percentile=90, device=device, max_images=args.calib_size)
            epsilon_used = float(epsilon_est)
            print(f"Chosen epsilon for PPW (based on 90th percentile) = {epsilon_used}")
        except Exception as e:
            print('Epsilon estimation failed, falling back to provided --epsilon value. Error:', e)
            epsilon_used = args.epsilon

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Images'):
        img_path = str(row[args.img_col])
        try:
            inp_t = preprocess_image(img_path, base_dir=args.base_dir, img_size=args.img_size)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        recon_t = reconstruct_image(model, inp_t, device=device)
        a = tensor_to_numpy_normalized(inp_t)
        b = tensor_to_numpy_normalized(recon_t)

        # ensure 2D arrays
        if a.ndim == 3:
            a = a.squeeze()
        if b.ndim == 3:
            b = b.squeeze()

        mse_v = compute_mse(a, b)
        ssim_v = compute_ssim(a, b)
        ppw_v = compute_ppw(a, b, epsilon=epsilon_used)
        edge_v = compute_edge_difference_normalized(a, b)

        results.append({'image_path': img_path, 'mse': mse_v, 'ssim': ssim_v, 'ppw': ppw_v, 'edge_diff_norm': edge_v, 'epsilon': epsilon_used})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print('Saved results to', args.output)


if __name__ == '__main__':
    main()
