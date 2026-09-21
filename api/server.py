import json
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from tokenizer.tokenizer import ByteBPETokenizer
from model.roblox_llm import RobloxLLM


ROOT = Path(__file__).resolve().parents[1]

app = Flask(__name__)

# Allow the GitHub Pages frontend to communicate with this API.
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


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

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu"
    )

    model_config = ckpt["config"]

    _model = RobloxLLM(
        vocab_size=ckpt["vocab_size"],
        n_embd=model_config["n_embd"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        block_size=model_config["block_size"],
        dropout=model_config["dropout"],
    )

    _model.load_state_dict(ckpt["model"])
    _model.eval()


@app.get("/")
def root():
    return jsonify({
        "name": "Roblox LLM Level 2",
        "status": "online"
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None
    })


@app.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}

    prompt = body.get("prompt", "")

    if not prompt:
        return jsonify({
            "error": "prompt is required"
        }), 400

    try:
        load_model()

        config_path = ROOT / "config.json"
        config = json.loads(config_path.read_text())

        tokens = _tokenizer.encode(prompt)

        idx = torch.tensor(
            [tokens],
            dtype=torch.long
        )

        max_new_tokens = body.get(
            "max_new_tokens",
            config["generation"]["max_new_tokens"]
        )

        temperature = body.get(
            "temperature",
            config["generation"]["temperature"]
        )

        top_k = body.get(
            "top_k",
            config["generation"]["top_k"]
        )

        with torch.no_grad():
            output = _model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        text = _tokenizer.decode(
            output[0].tolist()
        )

        return jsonify({
            "text": text
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
