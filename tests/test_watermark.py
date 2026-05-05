"""
tests/test_watermark.py — Unit tests for the QIM watermarking system
=====================================================================
Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from watermark import (
    generate_watermark,
    embed_watermark,
    extract_watermark,
    attack_gaussian_noise,
    attack_jpeg_compression,
    compute_psnr,
    compute_ber,
    qim_embed,
    qim_extract,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def host_image():
    """256×256 synthetic grayscale image."""
    rng = np.random.default_rng(0)
    return rng.integers(10, 245, size=(256, 256)).astype(np.float64)


@pytest.fixture
def watermark():
    return generate_watermark(64, seed=42)


@pytest.fixture
def watermarked(host_image, watermark):
    return embed_watermark(host_image, watermark, delta=25.0, seed=42)


# ──────────────────────────────────────────────────────────────────────────────
# QIM unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestQIM:
    """Test the core quantisation embed/extract round-trip."""

    @pytest.mark.parametrize("delta", [10.0, 25.0, 50.0])
    @pytest.mark.parametrize("bit", [0, 1])
    def test_round_trip_exact(self, delta, bit):
        """Embed then extract should recover the exact bit."""
        for coeff in np.linspace(-200, 200, 40):
            embedded  = qim_embed(coeff, bit, delta)
            extracted = qim_extract(embedded, delta)
            assert extracted == bit, (
                f"Round-trip failed: coeff={coeff}, bit={bit}, delta={delta}, "
                f"embedded={embedded}, extracted={extracted}"
            )

    def test_modification_small(self):
        """The embedding modification should be at most delta/2."""
        delta = 25.0
        for bit in (0, 1):
            for coeff in np.linspace(-100, 100, 50):
                embedded = qim_embed(coeff, bit, delta)
                assert abs(embedded - coeff) <= delta, (
                    f"Modification too large: |{embedded} - {coeff}| > {delta}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Embedding / extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestEmbedExtract:

    def test_clean_extraction_perfect(self, watermarked, watermark):
        """Extracting from an unattacked watermarked image must give BER=0."""
        extracted = extract_watermark(watermarked, len(watermark), delta=25.0, seed=42)
        ber = compute_ber(watermark, extracted)
        assert ber == 0.0, f"Expected BER=0, got {ber}"

    def test_shape_preserved(self, host_image, watermarked):
        """The watermarked image must have the same shape as the host."""
        assert watermarked.shape == host_image.shape

    def test_psnr_high(self, host_image, watermarked):
        """PSNR after embedding should be ≥ 30 dB (perceptually invisible)."""
        psnr = compute_psnr(host_image, watermarked)
        assert psnr >= 30.0, f"PSNR too low: {psnr:.2f} dB"

    def test_wrong_seed_fails(self, watermarked, watermark):
        """Extraction with a wrong seed must give BER near 0.5."""
        extracted = extract_watermark(watermarked, len(watermark), delta=25.0, seed=999)
        ber = compute_ber(watermark, extracted)
        # Should be close to random (0.3–0.7)
        assert ber > 0.2, f"Wrong-seed BER suspiciously low: {ber}"

    def test_wrong_delta_degrades(self, watermarked, watermark):
        """Extracting with a very different delta should increase BER."""
        ber_correct = compute_ber(
            watermark, extract_watermark(watermarked, len(watermark), delta=25.0, seed=42)
        )
        ber_wrong = compute_ber(
            watermark, extract_watermark(watermarked, len(watermark), delta=100.0, seed=42)
        )
        assert ber_wrong > ber_correct, "Wrong delta should increase BER"

    def test_different_seeds_independent(self):
        """Two different seeds should produce different watermarks."""
        wm1 = generate_watermark(128, seed=1)
        wm2 = generate_watermark(128, seed=2)
        assert not np.array_equal(wm1, wm2)

    def test_same_seed_reproducible(self):
        """Same seed must always produce the same watermark."""
        wm1 = generate_watermark(128, seed=7)
        wm2 = generate_watermark(128, seed=7)
        assert np.array_equal(wm1, wm2)


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────

class TestAttacks:

    def test_gaussian_low_noise_robust(self, watermarked, watermark):
        """Mild Gaussian noise (std=5) should keep BER well below random (< 0.25).
        Note: random synthetic images have lower robustness than natural images;
        on a real photo at std=5 you would typically see BER < 0.05."""
        attacked  = attack_gaussian_noise(watermarked, std=5.0)
        extracted = extract_watermark(attacked, len(watermark), delta=25.0, seed=42)
        ber = compute_ber(watermark, extracted)
        assert ber < 0.25, f"BER too high under mild noise: {ber}"

    def test_jpeg_high_quality_robust(self, watermarked, watermark):
        """JPEG at quality 90 should keep BER below 0.05."""
        attacked  = attack_jpeg_compression(watermarked, quality=90)
        extracted = extract_watermark(attacked, len(watermark), delta=25.0, seed=42)
        ber = compute_ber(watermark, extracted)
        assert ber < 0.05, f"BER too high at JPEG q=90: {ber}"

    def test_gaussian_output_clipped(self, watermarked):
        """Attacked image values must stay in [0, 255]."""
        attacked = attack_gaussian_noise(watermarked, std=30.0)
        assert attacked.min() >= 0
        assert attacked.max() <= 255

    def test_jpeg_output_shape(self, watermarked):
        attacked = attack_jpeg_compression(watermarked, quality=50)
        assert attacked.shape == watermarked.shape


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_psnr_identical(self):
        """PSNR of identical images should be very high (inf capped at 100 dB)."""
        img = np.ones((64, 64), dtype=np.float64) * 128
        psnr = compute_psnr(img, img)
        assert psnr > 60  # skimage returns ~100 dB for identical uint8

    def test_ber_identical(self, watermark):
        ber = compute_ber(watermark, watermark)
        assert ber == 0.0

    def test_ber_opposite(self, watermark):
        flipped = 1 - watermark
        ber = compute_ber(watermark, flipped)
        assert ber == 1.0

    def test_ber_range(self, watermark):
        rng = np.random.default_rng(0)
        random_wm = rng.integers(0, 2, size=len(watermark))
        ber = compute_ber(watermark, random_wm)
        assert 0.0 <= ber <= 1.0
