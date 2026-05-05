"""
app.py — Simple Flask web interface for the QIM watermarking system
====================================================================
Provides a basic web UI so Azure App Service has an HTTP server to display.
"""

from flask import Flask, render_template_string, jsonify
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>QIM Watermarking System</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f0f1a;
      color: #e0e0e0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .card {
      background: #1a1a2e;
      border: 1px solid #2a2a4a;
      border-radius: 12px;
      padding: 2.5rem 3rem;
      max-width: 680px;
      width: 100%;
      text-align: center;
    }
    h1 { font-size: 1.8rem; color: #00b4d8; margin-bottom: 0.5rem; }
    .subtitle { color: #888; font-size: 0.95rem; margin-bottom: 2rem; }
    .badge {
      display: inline-block;
      background: #0d3b4f;
      color: #00b4d8;
      border: 1px solid #00b4d8;
      border-radius: 20px;
      padding: 0.3rem 0.9rem;
      font-size: 0.8rem;
      margin: 0.25rem;
    }
    .section { margin: 1.8rem 0; text-align: left; }
    .section h2 { font-size: 1rem; color: #90e0ef; margin-bottom: 0.75rem; border-bottom: 1px solid #2a2a4a; padding-bottom: 0.4rem; }
    .feature { display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 0.6rem; }
    .dot { width: 8px; height: 8px; background: #00b4d8; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }
    .status {
      background: #0a2e1a;
      border: 1px solid #00c853;
      border-radius: 8px;
      padding: 0.75rem 1.25rem;
      color: #00c853;
      font-size: 0.9rem;
      margin-top: 1.5rem;
    }
    code {
      background: #111;
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-size: 0.85rem;
      color: #f4a261;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>🔐 QIM Watermarking System</h1>
    <p class="subtitle">Sécurisation des images 2D par tatouage numérique basé sur QIM</p>

    <div>
      <span class="badge">Python 3.12</span>
      <span class="badge">DCT + QIM</span>
      <span class="badge">Docker</span>
      <span class="badge">GitHub Actions</span>
      <span class="badge">Azure App Service</span>
    </div>

    <div class="section">
      <h2>What this system does</h2>
      <div class="feature"><div class="dot"></div><div>Embeds an invisible binary watermark into grayscale images using 2D DCT block processing</div></div>
      <div class="feature"><div class="dot"></div><div>Uses <strong>Quantization Index Modulation (QIM)</strong> on mid-frequency DCT coefficients</div></div>
      <div class="feature"><div class="dot"></div><div>Extracts the watermark using a secret key (seed + delta) — wrong key gives random bits</div></div>
      <div class="feature"><div class="dot"></div><div>Simulates attacks: Gaussian noise and JPEG compression</div></div>
      <div class="feature"><div class="dot"></div><div>Measures quality with <strong>PSNR</strong> and robustness with <strong>BER</strong></div></div>
    </div>

    <div class="section">
      <h2>CLI Usage</h2>
      <div class="feature"><div class="dot"></div><div><code>python cli.py run --image photo.png --bits 64 --delta 25</code></div></div>
      <div class="feature"><div class="dot"></div><div><code>python cli.py embed --image photo.png --output watermarked.png</code></div></div>
      <div class="feature"><div class="dot"></div><div><code>python cli.py extract --image watermarked.png --bits 64 --compare</code></div></div>
      <div class="feature"><div class="dot"></div><div><code>pytest tests/ -v</code> → 22 tests passing</div></div>
    </div>

    <div class="status">
      ✅ Deployed successfully via GitHub Actions CI/CD pipeline
    </div>
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "qim-watermarking"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)