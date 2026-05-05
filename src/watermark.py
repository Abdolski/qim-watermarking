"""
watermark.py — QIM-based digital watermarking using 2D DCT
=============================================================
Supports:
  - Embedding a binary watermark in DCT mid-frequency coefficients
  - Extracting the watermark from a (possibly attacked) image
  - Simulating attacks: Gaussian noise, JPEG compression
  - Quality metrics: PSNR, BER (Bit Error Rate)
"""

import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import cv2
import os
import json


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load a grayscale image and return as float64 array in [0, 255]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img.astype(np.float64)


def save_image(path: str, img: np.ndarray) -> None:
    """Save a float64 array as a grayscale PNG."""
    out = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


def generate_watermark(size: int, seed: int = 42) -> np.ndarray:
    """
    Generate a pseudo-random binary watermark of `size` bits.

    Parameters
    ----------
    size : int   Number of watermark bits.
    seed : int   Secret key for reproducibility.

    Returns
    -------
    np.ndarray of shape (size,) with values in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=size).astype(np.int32)


def _get_coeff_index(block_size: int, seed: int) -> tuple[int, int]:
    """
    Select ONE pseudo-random mid-frequency DCT coefficient position for a block.
    Mid-frequency = positions (r, c) where 2 <= r+c <= block_size-2.
    The same position is used for every block; the block assignment provides variety.
    """
    rng = np.random.default_rng(seed + 1)
    candidates = [
        (r, c)
        for r in range(block_size)
        for c in range(block_size)
        if 2 <= r + c <= block_size - 2
    ]
    idx = rng.integers(0, len(candidates))
    return candidates[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Core QIM functions
# ──────────────────────────────────────────────────────────────────────────────

def qim_embed(coeff: float, bit: int, delta: float) -> float:
    """
    Embed one watermark bit into one DCT coefficient using QIM.

    QIM principle:
      - If bit == 0: quantise to the nearest even multiple of delta/2
      - If bit == 1: quantise to the nearest odd  multiple of delta/2

    Parameters
    ----------
    coeff : float   Original DCT coefficient value.
    bit   : int     Watermark bit (0 or 1).
    delta : float   Quantisation step (controls robustness vs. quality).

    Returns
    -------
    Modified coefficient.
    """
    half = delta / 2.0
    q = round(coeff / half)
    if (q % 2) != bit:
        if coeff > q * half:
            q += 1
        else:
            q -= 1
    return q * half


def qim_extract(coeff: float, delta: float) -> int:
    """
    Extract one watermark bit from a (possibly attacked) DCT coefficient.

    Returns the parity of round(coeff / (delta/2)).
    """
    half = delta / 2.0
    return int(round(coeff / half)) % 2


# ──────────────────────────────────────────────────────────────────────────────
# Embed
# ──────────────────────────────────────────────────────────────────────────────

def embed_watermark(
    host: np.ndarray,
    watermark: np.ndarray,
    delta: float = 25.0,
    block_size: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """
    Embed a binary watermark into the host image using block DCT + QIM.

    Parameters
    ----------
    host      : 2-D float64 array (grayscale image, values 0–255).
    watermark : 1-D int array of bits {0, 1}.
    delta     : QIM step — larger ⟹ more robust but more visible.
    block_size: DCT block size (typically 8).
    seed      : Secret key for coefficient selection.

    Returns
    -------
    Watermarked image as float64 array.
    """
    h, w = host.shape
    n_bits = len(watermark)

    # Compute how many complete blocks fit
    n_blocks_r = h // block_size
    n_blocks_c = w // block_size
    total_blocks = n_blocks_r * n_blocks_c

    if total_blocks < n_bits:
        raise ValueError(
            f"Image too small: need {n_bits} blocks but only {total_blocks} available."
        )

    # Pseudo-random block ordering (permutation of block indices) — acts as secret key
    rng = np.random.default_rng(seed + 99)
    block_order = rng.permutation(total_blocks)[:n_bits]

    # One fixed mid-frequency coefficient position (same for all blocks)
    r_coeff, c_coeff = _get_coeff_index(block_size, seed)

    watermarked = host.copy()

    for bit_i, block_i in enumerate(block_order):
        br = (block_i // n_blocks_c) * block_size
        bc = (block_i % n_blocks_c) * block_size
        block = watermarked[br : br + block_size, bc : bc + block_size]
        dct_block = dctn(block, norm="ortho")
        dct_block[r_coeff, c_coeff] = qim_embed(dct_block[r_coeff, c_coeff], watermark[bit_i], delta)
        watermarked[br : br + block_size, bc : bc + block_size] = idctn(
            dct_block, norm="ortho"
        )

    return watermarked


# ──────────────────────────────────────────────────────────────────────────────
# Extract
# ──────────────────────────────────────────────────────────────────────────────

def extract_watermark(
    watermarked: np.ndarray,
    n_bits: int,
    delta: float = 25.0,
    block_size: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract the binary watermark from a (possibly attacked) image.

    Parameters
    ----------
    watermarked : 2-D float64 grayscale image.
    n_bits      : Expected number of watermark bits.
    delta       : QIM step used during embedding (must match).
    block_size  : Block size used during embedding (must match).
    seed        : Secret key used during embedding (must match).

    Returns
    -------
    Extracted watermark as 1-D int array of bits {0, 1}.
    """
    h, w = watermarked.shape
    n_blocks_r = h // block_size
    n_blocks_c = w // block_size
    total_blocks = n_blocks_r * n_blocks_c

    rng = np.random.default_rng(seed + 99)
    block_order = rng.permutation(total_blocks)[:n_bits]
    r_coeff, c_coeff = _get_coeff_index(block_size, seed)

    extracted = np.zeros(n_bits, dtype=np.int32)
    for bit_i, block_i in enumerate(block_order):
        br = (block_i // n_blocks_c) * block_size
        bc = (block_i % n_blocks_c) * block_size
        block = watermarked[br : br + block_size, bc : bc + block_size]
        dct_block = dctn(block, norm="ortho")
        extracted[bit_i] = qim_extract(dct_block[r_coeff, c_coeff], delta)

    return extracted


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (higher = better quality)."""
    orig_u8 = np.clip(original, 0, 255).astype(np.uint8)
    mod_u8  = np.clip(modified, 0, 255).astype(np.uint8)
    return float(psnr_metric(orig_u8, mod_u8, data_range=255))


def compute_ber(original_wm: np.ndarray, extracted_wm: np.ndarray) -> float:
    """Bit Error Rate — fraction of bits that differ (0.0 = perfect)."""
    return float(np.mean(original_wm != extracted_wm))


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────

def attack_gaussian_noise(img: np.ndarray, std: float = 10.0, seed: int = 0) -> np.ndarray:
    """Add additive Gaussian noise with the given standard deviation."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255)


def attack_jpeg_compression(img: np.ndarray, quality: int = 50) -> np.ndarray:
    """Simulate JPEG compression at the given quality level (1–100)."""
    uint8 = np.clip(img, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", uint8, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    image_path: str,
    n_bits: int = 64,
    delta: float = 25.0,
    block_size: int = 8,
    seed: int = 42,
    output_dir: str = "results",
) -> dict:
    """
    Complete watermarking pipeline:
      1. Load image
      2. Generate watermark
      3. Embed → save watermarked image
      4. Extract from clean watermarked image
      5. Apply attacks and evaluate

    Returns a dict with all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    host = load_image(image_path)
    print(f"[+] Host image loaded: {host.shape[0]}×{host.shape[1]} px")

    # 2. Watermark
    wm = generate_watermark(n_bits, seed=seed)
    print(f"[+] Watermark generated: {n_bits} bits, seed={seed}")

    # 3. Embed
    watermarked = embed_watermark(host, wm, delta=delta, block_size=block_size, seed=seed)
    wm_path = os.path.join(output_dir, "watermarked.png")
    save_image(wm_path, watermarked)
    psnr_embed = compute_psnr(host, watermarked)
    print(f"[+] Embedded watermark  →  PSNR = {psnr_embed:.2f} dB")

    results = {
        "image": image_path,
        "n_bits": n_bits,
        "delta": delta,
        "seed": seed,
        "psnr_after_embed": round(psnr_embed, 4),
    }

    # 4. Clean extraction
    extracted_clean = extract_watermark(watermarked, n_bits, delta=delta, block_size=block_size, seed=seed)
    ber_clean = compute_ber(wm, extracted_clean)
    print(f"[+] Clean extraction    →  BER = {ber_clean:.4f}")
    results["ber_clean"] = round(ber_clean, 6)

    # 5. Attacks
    attacks = {
        "gaussian_noise_std10":  attack_gaussian_noise(watermarked, std=10.0),
        "gaussian_noise_std25":  attack_gaussian_noise(watermarked, std=25.0),
        "jpeg_quality_75":       attack_jpeg_compression(watermarked, quality=75),
        "jpeg_quality_50":       attack_jpeg_compression(watermarked, quality=50),
        "jpeg_quality_25":       attack_jpeg_compression(watermarked, quality=25),
    }

    results["attacks"] = {}
    for attack_name, attacked_img in attacks.items():
        save_image(os.path.join(output_dir, f"{attack_name}.png"), attacked_img)
        extracted = extract_watermark(attacked_img, n_bits, delta=delta, block_size=block_size, seed=seed)
        ber = compute_ber(wm, extracted)
        psnr_atk = compute_psnr(watermarked, attacked_img)
        results["attacks"][attack_name] = {
            "psnr": round(psnr_atk, 4),
            "ber":  round(ber, 6),
        }
        print(f"    {attack_name:<30}  PSNR={psnr_atk:.2f} dB   BER={ber:.4f}")

    # Save JSON report
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Report saved → {report_path}")

    return results
