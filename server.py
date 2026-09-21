import json
from pathlib import Path
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from tokenizer.tokenizer import ByteBPETokenizer
from model.roblox_llm import RobloxLLM

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__)
CORS(app)

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    ckpt_path = ROOT / "checkpoints" / "roblox_level2_llm.pt"
    tok_path = ROOT / "tokenizer" / "roblox_bpe_tokenizer.json"

    if not ckpt_path.exists():
        raise FileNotFoundError("Model checkpoint is missing.")
    _tokenizer = ByteBPETokenizer.load(tok_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    m = ckpt["config"]

    _model = RobloxLLM(
        vocab_size=ckpt["vocab_size"],
        n_embd=m["n_embd"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        block_size=m["block_size"],
        dropout=m["dropout"]
    )
    _model.load_state_dict(ckpt["model"])
    _model.eval()

@app.get("/")
def root():
    return jsonify({"name": "Roblox LLM Level 2", "status": "online"})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})

@app.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        load_model()
        cfg = json.loads((ROOT / "config.json").read_text())
        idx = torch.tensor([_tokenizer.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            out = _model.generate(
                idx,
                max_new_tokens=body.get("max_new_tokens", cfg["generation"]["max_new_tokens"]),
                temperature=body.get("temperature", cfg["generation"]["temperature"]),
                top_k=body.get("top_k", cfg["generation"]["top_k"])
            )
        return jsonify({"text": _tokenizer.decode(out[0].tolist())})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
