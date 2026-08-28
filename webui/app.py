"""Standalone test UI for the Stage 2 head: drag an image in, see its
horseness score and whether it would pass the gate.

Deliberately decoupled from the rest of the project -- it only *imports*
from horse/ (embed, model, gate), it doesn't get imported back. Everything
this needs lives in this one folder (app.py + requirements.txt), so
deleting webui/ removes it cleanly with nothing else to clean up.

Run:
    pip install -r webui/requirements.txt
    python webui/app.py [--head horse/data/head.pt] [--threshold 2.5] [--port 5000]

Then open http://127.0.0.1:5000 and drop an image in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Let `python webui/app.py` find the horse/ package regardless of cwd,
# without needing this folder to be a package itself (`python -m`) or a
# PYTHONPATH set by hand -- one line, easy to spot when webui/ is deleted.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from flask import Flask, jsonify, request
from PIL import Image

from horse.embed import ClipEmbedder
from horse.gate import DEFAULT_THRESHOLD, passes_gate
from horse.model import HorseHead

app = Flask(__name__)

# Populated once in main() before app.run() -- a single frozen encoder and
# head shared across requests, so scoring stays cheap per upload.
STATE: dict = {"embedder": None, "head": None, "threshold": DEFAULT_THRESHOLD, "head_loaded": False, "head_path": None}


def render_index() -> str:
    if STATE["head_loaded"]:
        banner = ""
    else:
        banner = (
            '<div class="banner">'
            f"No trained checkpoint found at <code>{STATE['head_path']}</code> — "
            "using an untrained (random) head. Scores below are meaningless until "
            "you run <code>python -m horse.train</code>."
            "</div>"
        )
    return INDEX_TEMPLATE.replace("{{banner}}", banner).replace("{{threshold}}", str(STATE["threshold"]))


@app.get("/")
def index():
    return render_index()


@app.post("/score")
def score():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no image uploaded"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- surface any bad-upload reason to the UI
        return jsonify({"error": f"could not read image: {exc}"}), 400

    embedding = STATE["embedder"].embed_image(image)
    with torch.no_grad():
        score_value = STATE["head"](torch.from_numpy(embedding).unsqueeze(0)).item()

    threshold = STATE["threshold"]
    return jsonify(
        {
            "score": score_value,
            "threshold": threshold,
            "passed": passes_gate(score_value, threshold),
            "head_loaded": STATE["head_loaded"],
        }
    )


INDEX_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Horse Stage 2 — test UI</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 560px; margin: 48px auto; color: #1a1a1a; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  p.sub { color: #666; margin-top: 0; }
  .banner { background: #fff3cd; border: 1px solid #ffdd80; padding: 10px 14px; border-radius: 6px; margin-bottom: 20px; font-size: 0.9rem; }
  #drop {
    border: 2px dashed #aaa; border-radius: 10px; padding: 40px 20px;
    text-align: center; color: #666; cursor: pointer; transition: border-color 0.15s, background 0.15s;
  }
  #drop.drag { border-color: #4a7; background: #f3fdf6; }
  #preview { max-width: 100%; max-height: 260px; margin-top: 16px; border-radius: 8px; display: none; }
  #result { margin-top: 20px; display: none; }
  .score { font-size: 2.2rem; font-weight: 700; }
  .verdict { display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; margin-left: 10px; vertical-align: middle; }
  .pass { background: #d6f5e3; color: #146c3a; }
  .reject { background: #fde0e0; color: #a11d1d; }
  .meta { color: #666; font-size: 0.85rem; margin-top: 6px; }
  #error { color: #a11d1d; margin-top: 16px; display: none; }
  input[type=file] { display: none; }
</style>
</head>
<body>
  <h1>🐴 Stage 2 test UI</h1>
  <p class="sub">Drop an image to score it. Gate threshold: <strong>{{threshold}}</strong></p>
  {{banner}}
  <div id="drop">Drag an image here, or click to choose one</div>
  <input type="file" id="fileInput" accept="image/*">
  <img id="preview">
  <div id="result">
    <span class="score" id="scoreValue"></span>
    <span class="verdict" id="verdict"></span>
    <div class="meta" id="meta"></div>
  </div>
  <div id="error"></div>

<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const result = document.getElementById('result');
const scoreValue = document.getElementById('scoreValue');
const verdict = document.getElementById('verdict');
const meta = document.getElementById('meta');
const errorBox = document.getElementById('error');

drop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

['dragenter', 'dragover'].forEach(evt =>
  drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.add('drag'); })
);
['dragleave', 'drop'].forEach(evt =>
  drop.addEventListener(evt, e => { e.preventDefault(); drop.classList.remove('drag'); })
);
drop.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  errorBox.style.display = 'none';
  result.style.display = 'none';

  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';

  const formData = new FormData();
  formData.append('image', file);

  fetch('/score', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        errorBox.textContent = data.error;
        errorBox.style.display = 'block';
        return;
      }
      scoreValue.textContent = data.score.toFixed(3);
      verdict.textContent = data.passed ? 'PASS' : 'REJECT';
      verdict.className = 'verdict ' + (data.passed ? 'pass' : 'reject');
      meta.textContent = `threshold ${data.threshold}` + (data.head_loaded ? '' : ' — untrained head, ignore this score');
      result.style.display = 'block';
    })
    .catch(err => {
      errorBox.textContent = 'request failed: ' + err;
      errorBox.style.display = 'block';
    });
}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone drag-and-drop test UI for the Stage 2 head.")
    parser.add_argument("--head", default="horse/data/head.pt", help="Path to a trained head checkpoint.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    STATE["embedder"] = ClipEmbedder()
    STATE["threshold"] = args.threshold
    STATE["head_path"] = args.head

    if Path(args.head).exists():
        STATE["head"] = HorseHead.load(args.head)
        STATE["head_loaded"] = True
    else:
        print(f"[webui] no checkpoint at {args.head} -- using an untrained (random) head; scores are meaningless.")
        STATE["head"] = HorseHead()
        STATE["head"].eval()

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
