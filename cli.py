#!/usr/bin/env python3
"""
cli.py — Command-Line Interface for the QIM watermarking system
================================================================

Usage examples
--------------
# Full pipeline on a single image
python cli.py run --image samples/lena.png --bits 64 --delta 25

# Embed only
python cli.py embed --image samples/lena.png --output results/watermarked.png

# Extract only
python cli.py extract --image results/watermarked.png --bits 64

# Attack simulation
python cli.py attack --image results/watermarked.png --type jpeg --quality 50

# Print results summary
python cli.py report --file results/report.json
"""

import argparse
import sys
import os
import json

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from watermark import (
    load_image,
    save_image,
    generate_watermark,
    embed_watermark,
    extract_watermark,
    attack_gaussian_noise,
    attack_jpeg_compression,
    compute_psnr,
    compute_ber,
    run_full_pipeline,
)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_run(args):
    """Run the complete pipeline and print a summary table."""
    results = run_full_pipeline(
        image_path=args.image,
        n_bits=args.bits,
        delta=args.delta,
        block_size=args.block_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"  PSNR after embedding : {results['psnr_after_embed']} dB")
    print(f"  BER clean extraction : {results['ber_clean']}")
    print()
    print(f"  {'Attack':<32} {'PSNR (dB)':>10}  {'BER':>8}")
    print(f"  {'──────':<32} {'─────────':>10}  {'───':>8}")
    for name, m in results["attacks"].items():
        print(f"  {name:<32} {m['psnr']:>10.2f}  {m['ber']:>8.4f}")
    print("────────────────────────────────────────────────────────────")


def cmd_embed(args):
    host = load_image(args.image)
    wm   = generate_watermark(args.bits, seed=args.seed)
    watermarked = embed_watermark(host, wm, delta=args.delta, seed=args.seed)
    save_image(args.output, watermarked)
    psnr_val = compute_psnr(host, watermarked)
    print(f"[+] Watermarked image saved → {args.output}")
    print(f"[+] PSNR = {psnr_val:.2f} dB")


def cmd_extract(args):
    img = load_image(args.image)
    extracted = extract_watermark(img, args.bits, delta=args.delta, seed=args.seed)
    print(f"[+] Extracted {args.bits} bits (seed={args.seed})")
    print("    Bits:", "".join(map(str, extracted)))
    if args.compare:
        original = generate_watermark(args.bits, seed=args.seed)
        ber = compute_ber(original, extracted)
        print(f"[+] BER vs original watermark : {ber:.4f}")


def cmd_attack(args):
    img = load_image(args.image)
    if args.type == "gaussian":
        attacked = attack_gaussian_noise(img, std=args.std)
        label = f"gaussian_std{int(args.std)}"
    elif args.type == "jpeg":
        attacked = attack_jpeg_compression(img, quality=args.quality)
        label = f"jpeg_q{args.quality}"
    else:
        print(f"Unknown attack type: {args.type}")
        sys.exit(1)

    out_path = args.output or f"results/attacked_{label}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image(out_path, attacked)
    psnr_val = compute_psnr(img, attacked)
    print(f"[+] Attack '{label}' applied → {out_path}")
    print(f"[+] PSNR = {psnr_val:.2f} dB")

    if args.extract_bits:
        extracted = extract_watermark(attacked, args.extract_bits, delta=args.delta, seed=args.seed)
        original  = generate_watermark(args.extract_bits, seed=args.seed)
        ber = compute_ber(original, extracted)
        print(f"[+] BER after attack : {ber:.4f}")


def cmd_report(args):
    with open(args.file) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qim-watermark",
        description="QIM-based digital image watermarking (DCT domain)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args factory
    def add_common(p):
        p.add_argument("--bits",       type=int,   default=64,    help="Number of watermark bits")
        p.add_argument("--delta",      type=float, default=25.0,  help="QIM step size")
        p.add_argument("--seed",       type=int,   default=42,    help="Secret key (seed)")
        p.add_argument("--block-size", type=int,   default=8,     dest="block_size", help="DCT block size")

    # run
    p_run = sub.add_parser("run", help="Full pipeline: embed + attacks + report")
    p_run.add_argument("--image",      required=True,                    help="Path to host image")
    p_run.add_argument("--output-dir", default="results", dest="output_dir")
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    # embed
    p_emb = sub.add_parser("embed", help="Embed watermark into image")
    p_emb.add_argument("--image",  required=True, help="Host image path")
    p_emb.add_argument("--output", default="results/watermarked.png")
    add_common(p_emb)
    p_emb.set_defaults(func=cmd_embed)

    # extract
    p_ext = sub.add_parser("extract", help="Extract watermark from image")
    p_ext.add_argument("--image",   required=True, help="Watermarked (or attacked) image")
    p_ext.add_argument("--compare", action="store_true", help="Compare with original watermark")
    add_common(p_ext)
    p_ext.set_defaults(func=cmd_extract)

    # attack
    p_atk = sub.add_parser("attack", help="Apply an attack to a watermarked image")
    p_atk.add_argument("--image",        required=True)
    p_atk.add_argument("--type",         choices=["gaussian", "jpeg"], required=True)
    p_atk.add_argument("--std",          type=float, default=10.0,  help="Gaussian noise std")
    p_atk.add_argument("--quality",      type=int,   default=50,    help="JPEG quality (1-100)")
    p_atk.add_argument("--output",       default=None)
    p_atk.add_argument("--extract-bits", type=int,   default=None, dest="extract_bits",
                        help="If set, extract and evaluate BER after attack")
    add_common(p_atk)
    p_atk.set_defaults(func=cmd_attack)

    # report
    p_rep = sub.add_parser("report", help="Print a JSON report")
    p_rep.add_argument("--file", default="results/report.json")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
