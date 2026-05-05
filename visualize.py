"""
visualize.py — Plotting utilities for the QIM watermarking project
==================================================================
Generates:
  - Side-by-side image comparison (original vs watermarked vs attacked)
  - BER vs attack strength curves
  - PSNR vs delta (step size) trade-off curve
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from watermark import (
    load_image,
    generate_watermark,
    embed_watermark,
    extract_watermark,
    attack_gaussian_noise,
    attack_jpeg_compression,
    compute_psnr,
    compute_ber,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Image comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_image_comparison(host, watermarked, attacked_dict, output_path="results/comparison.png"):
    """
    Show the host, watermarked, and each attacked version side-by-side.
    """
    n = 2 + len(attacked_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor("#1a1a2e")

    def _show(ax, img, title, psnr=None):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        label = title if psnr is None else f"{title}\nPSNR={psnr:.1f} dB"
        ax.set_title(label, color="white", fontsize=9, pad=6)
        ax.axis("off")

    _show(axes[0], host, "Host (original)")
    psnr_wm = compute_psnr(host, watermarked)
    _show(axes[1], watermarked, "Watermarked", psnr=psnr_wm)

    for i, (name, img) in enumerate(attacked_dict.items()):
        psnr_atk = compute_psnr(watermarked, img)
        _show(axes[2 + i], img, name.replace("_", "\n"), psnr=psnr_atk)

    plt.suptitle("QIM Watermarking — Image Comparison", color="white", fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved comparison → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. BER vs attack strength
# ──────────────────────────────────────────────────────────────────────────────

def plot_ber_vs_attack(
    image_path: str,
    n_bits: int = 64,
    delta: float = 25.0,
    seed: int = 42,
    output_path: str = "results/ber_vs_attack.png",
):
    host = load_image(image_path)
    wm   = generate_watermark(n_bits, seed=seed)
    watermarked = embed_watermark(host, wm, delta=delta, seed=seed)

    # Gaussian noise
    stds = np.arange(0, 55, 5)
    bers_gauss = []
    for std in stds:
        attacked = attack_gaussian_noise(watermarked, std=float(std))
        ext = extract_watermark(attacked, n_bits, delta=delta, seed=seed)
        bers_gauss.append(compute_ber(wm, ext))

    # JPEG quality
    qualities = list(range(10, 101, 5))
    bers_jpeg = []
    for q in qualities:
        attacked = attack_jpeg_compression(watermarked, quality=q)
        ext = extract_watermark(attacked, n_bits, delta=delta, seed=seed)
        bers_jpeg.append(compute_ber(wm, ext))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax1.plot(stds, bers_gauss, marker="o", color="#00b4d8", linewidth=2)
    ax1.axhline(0.5, color="#e63946", linestyle="--", linewidth=1.2, label="Random (BER=0.5)")
    ax1.set_xlabel("Gaussian Noise std")
    ax1.set_ylabel("BER")
    ax1.set_title("BER vs Gaussian Noise")
    ax1.set_ylim(-0.02, 0.6)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax1.grid(True, color="#2a2a4a", linewidth=0.7)

    ax2.plot(qualities, bers_jpeg, marker="s", color="#90e0ef", linewidth=2)
    ax2.axhline(0.5, color="#e63946", linestyle="--", linewidth=1.2, label="Random (BER=0.5)")
    ax2.set_xlabel("JPEG Quality (%)")
    ax2.set_ylabel("BER")
    ax2.set_title("BER vs JPEG Compression Quality")
    ax2.set_ylim(-0.02, 0.6)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax2.grid(True, color="#2a2a4a", linewidth=0.7)

    plt.suptitle(f"Robustness Analysis  (δ={delta}, n={n_bits} bits)", color="white", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved BER curves → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. PSNR vs delta trade-off
# ──────────────────────────────────────────────────────────────────────────────

def plot_psnr_vs_delta(
    image_path: str,
    n_bits: int = 64,
    seed: int = 42,
    output_path: str = "results/psnr_vs_delta.png",
):
    host = load_image(image_path)
    wm   = generate_watermark(n_bits, seed=seed)

    deltas = np.arange(5, 105, 5)
    psnrs  = []
    bers_jpeg50 = []

    for d in deltas:
        watermarked = embed_watermark(host, wm, delta=float(d), seed=seed)
        psnrs.append(compute_psnr(host, watermarked))
        # Robustness: JPEG q=50
        attacked = attack_jpeg_compression(watermarked, quality=50)
        ext = extract_watermark(attacked, n_bits, delta=float(d), seed=seed)
        bers_jpeg50.append(compute_ber(wm, ext))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    ax1.title.set_color("white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")

    color_psnr = "#00b4d8"
    color_ber  = "#f4a261"

    l1, = ax1.plot(deltas, psnrs, color=color_psnr, marker="o", linewidth=2, label="PSNR (dB)")
    ax1.set_xlabel("QIM step δ (delta)")
    ax1.set_ylabel("PSNR (dB)", color=color_psnr)
    ax1.tick_params(axis="y", labelcolor=color_psnr)
    ax1.grid(True, color="#2a2a4a", linewidth=0.7)

    ax2 = ax1.twinx()
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    ax2.yaxis.label.set_color("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    l2, = ax2.plot(deltas, bers_jpeg50, color=color_ber, marker="s", linewidth=2,
                   linestyle="--", label="BER (JPEG q=50)")
    ax2.set_ylabel("BER after JPEG q=50", color=color_ber)
    ax2.tick_params(axis="y", labelcolor=color_ber)
    ax2.set_ylim(-0.02, 0.6)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    ax1.set_title("Quality vs Robustness Trade-off", color="white", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved PSNR vs δ → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Watermark bit comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_watermark_bits(
    original_wm: np.ndarray,
    extracted_wm: np.ndarray,
    output_path: str = "results/wm_bits.png",
    max_show: int = 64,
):
    n = min(len(original_wm), max_show)
    orig = original_wm[:n]
    ext  = extracted_wm[:n]
    errors = (orig != ext).astype(int)

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    x = np.arange(n)
    ax.bar(x[errors == 0], orig[errors == 0], color="#00b4d8", width=0.6, label="Correct bit")
    ax.bar(x[errors == 1], orig[errors == 1], color="#e63946", width=0.6, label="Error")
    ax.step(x - 0.3, ext, where="post", color="#f4a261", linewidth=1.5, label="Extracted")

    ber = compute_ber(orig, ext)
    ax.set_title(f"Watermark bits comparison  (BER = {ber:.4f})", color="white")
    ax.set_xlabel("Bit index", color="white")
    ax.set_ylabel("Bit value", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.grid(True, color="#2a2a4a", linewidth=0.5, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved watermark bits chart → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo — run this file directly to generate all plots
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate all visualizations")
    parser.add_argument("--image", required=True, help="Host image path")
    parser.add_argument("--bits",  type=int, default=64)
    parser.add_argument("--delta", type=float, default=25.0)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--out",   default="results")
    args = parser.parse_args()

    host = load_image(args.image)
    wm   = generate_watermark(args.bits, seed=args.seed)
    watermarked = embed_watermark(host, wm, delta=args.delta, seed=args.seed)

    attacked_dict = {
        "Gaussian\nstd=10": attack_gaussian_noise(watermarked, std=10),
        "JPEG\nq=75":       attack_jpeg_compression(watermarked, quality=75),
        "JPEG\nq=50":       attack_jpeg_compression(watermarked, quality=50),
    }

    plot_image_comparison(host, watermarked, attacked_dict,
                          output_path=os.path.join(args.out, "comparison.png"))
    plot_ber_vs_attack(args.image, args.bits, args.delta, args.seed,
                       output_path=os.path.join(args.out, "ber_vs_attack.png"))
    plot_psnr_vs_delta(args.image, args.bits, args.seed,
                       output_path=os.path.join(args.out, "psnr_vs_delta.png"))

    # Bit chart for JPEG q=50
    attacked_50 = attack_jpeg_compression(watermarked, quality=50)
    ext_50 = extract_watermark(attacked_50, args.bits, delta=args.delta, seed=args.seed)
    plot_watermark_bits(wm, ext_50, output_path=os.path.join(args.out, "wm_bits_jpeg50.png"))
