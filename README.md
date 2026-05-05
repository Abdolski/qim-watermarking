# QIM Digital Image Watermarking

**Sécurisation des images 2D par tatouage numérique basé sur QIM**

Mini-projet L2-IRS 25/26 — Conception et implémentation d'un système de tatouage numérique robuste basé sur la méthode QIM en Python.

---

## Concept

This system embeds an invisible binary watermark inside a grayscale image using:

- **2D Discrete Cosine Transform (DCT)** — applied block by block (8×8)
- **Quantization Index Modulation (QIM)** — embeds each watermark bit by quantising one selected DCT coefficient
- **Mid-frequency coefficient selection** — via a secret pseudo-random key, giving a good invisibility/robustness trade-off
- **Attack simulation** — Gaussian noise and JPEG compression
- **Metrics** — PSNR (quality) and BER (robustness)

### Why DCT?

| Property | Benefit |
|----------|---------|
| Good JPEG robustness | Same domain as JPEG compression |
| Invisibility / robustness compromise | Mid-frequencies are perceptually less sensitive |
| Widely studied | Standard in multimedia watermarking literature |

---

## Project Structure

```
qim_watermarking/
├── src/
│   └── watermark.py       ← Core library (QIM, DCT, attacks, metrics)
├── tests/
│   └── test_watermark.py  ← Pytest unit tests
├── cli.py                 ← Command-line interface
├── visualize.py           ← Plotting & analysis charts
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.10+**.

---

## Quick Start

### 1 — Full pipeline (embed + attack + evaluate)

```bash
python cli.py run --image path/to/image.png --bits 64 --delta 25
```

Output in `results/`:
- `watermarked.png` — image with embedded watermark
- `gaussian_noise_std10.png`, `jpeg_quality_50.png`, … — attacked versions
- `report.json` — full metrics (PSNR, BER per attack)

---

### 2 — Embed only

```bash
python cli.py embed \
    --image  samples/lena.png \
    --output results/watermarked.png \
    --bits   64 \
    --delta  25 \
    --seed   42
```

---

### 3 — Extract only

```bash
python cli.py extract \
    --image   results/watermarked.png \
    --bits    64 \
    --seed    42 \
    --compare          # compare with original and print BER
```

---

### 4 — Simulate attacks

```bash
# Gaussian noise
python cli.py attack --image results/watermarked.png \
    --type gaussian --std 10 --extract-bits 64

# JPEG compression
python cli.py attack --image results/watermarked.png \
    --type jpeg --quality 50 --extract-bits 64
```

---

### 5 — Generate all analysis plots

```bash
python visualize.py --image samples/lena.png --bits 64 --delta 25
```

Produces in `results/`:
- `comparison.png` — side-by-side visual comparison
- `ber_vs_attack.png` — BER curves vs noise std and JPEG quality
- `psnr_vs_delta.png` — quality vs robustness trade-off
- `wm_bits_jpeg50.png` — bit-level extraction accuracy chart

---

### 6 — Run tests

```bash
pytest tests/ -v
```

---

## API Reference

### `src/watermark.py`

| Function | Description |
|----------|-------------|
| `generate_watermark(size, seed)` | Generate a pseudo-random binary watermark |
| `embed_watermark(host, watermark, delta, block_size, seed)` | Embed watermark via DCT + QIM |
| `extract_watermark(img, n_bits, delta, block_size, seed)` | Extract watermark bits |
| `attack_gaussian_noise(img, std)` | Simulate additive Gaussian noise |
| `attack_jpeg_compression(img, quality)` | Simulate JPEG compression |
| `compute_psnr(original, modified)` | Peak Signal-to-Noise Ratio (dB) |
| `compute_ber(original_wm, extracted_wm)` | Bit Error Rate |
| `run_full_pipeline(image_path, ...)` | Complete embed + attack + evaluate pipeline |

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `delta` | `25.0` | QIM step — larger = more robust but more visible |
| `n_bits` | `64` | Watermark length in bits |
| `seed` | `42` | Secret key — must match for extraction |
| `block_size` | `8` | DCT block size (standard = 8) |

---

## Functional Requirements — Mapping

| Requirement | Implementation |
|-------------|---------------|
| Charger une image | `load_image()` in `watermark.py` |
| Générer un watermark binaire | `generate_watermark()` |
| Insérer watermark via QIM | `embed_watermark()` with `qim_embed()` |
| Extraire watermark | `extract_watermark()` with `qim_extract()` |
| Attaque bruit gaussien | `attack_gaussian_noise()` |
| Attaque compression JPEG | `attack_jpeg_compression()` |
| Calculer PSNR et BER | `compute_psnr()`, `compute_ber()` |
| Interface simple (CLI) | `cli.py` |
| Résultats reproductibles | Deterministic `seed` parameter everywhere |
| Temps d'exécution < 5 sec | Single-image pipeline < 2 sec on modern hardware |

---

## Example Output

```
[+] Host image loaded: 512×512 px
[+] Watermark generated: 64 bits, seed=42
[+] Embedded watermark  →  PSNR = 41.23 dB
[+] Clean extraction    →  BER = 0.0000

    gaussian_noise_std10           PSNR= 28.13 dB   BER=0.0000
    gaussian_noise_std25           PSNR= 20.17 dB   BER=0.0313
    jpeg_quality_75                PSNR= 37.22 dB   BER=0.0000
    jpeg_quality_50                PSNR= 33.84 dB   BER=0.0156
    jpeg_quality_25                PSNR= 29.61 dB   BER=0.0625
```

A PSNR ≥ 40 dB indicates the watermark is perceptually invisible. A BER of 0 means perfect watermark recovery.
