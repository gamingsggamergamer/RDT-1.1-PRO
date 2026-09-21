import json
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from tokenizer.tokenizer import ByteBPETokenizer
from model.roblox_llm import RobloxLLM


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


_model = None
_tokenizer = None


def load_model():
    global _model
    global _tokenizer

    if _model is not None:
        return

    checkpoint_path = ROOT / "checkpoints" / "roblox_level2_llm.pt"
    tokenizer_path = ROOT / "tokenizer" / "roblox_bpe_tokenizer.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}"
        )

    _tokenizer = ByteBPETokenizer.load(tokenizer_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model_config = checkpoint["config"]

    _model = RobloxLLM(
        vocab_size=checkpoint["vocab_size"],
        n_embd=model_config["n_embd"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        block_size=model_config["block_size"],
        dropout=model_config["dropout"],
    )

    _model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    _model.eval()


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "Roblox LLM Level 2",
        "status": "online"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None
    })


@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(silent=True) or {}

    prompt = body.get("prompt", "")

    if not isinstance(prompt, str):
        return jsonify({
            "error": "prompt must be a string"
        }), 400

    prompt = prompt.strip()

    if not prompt:
        return jsonify({
            "error": "prompt is required"
        }), 400

    try:
        load_model()

        config_path = ROOT / "config.json"

        config = json.loads(
            config_path.read_text()
        )

        generation_config = config["generation"]

        max_new_tokens = int(
            body.get(
                "max_new_tokens",
                generation_config["max_new_tokens"]
            )
        )

        temperature = float(
            body.get(
                "temperature",
                generation_config["temperature"]
            )
        )

        top_k = int(
            body.get(
                "top_k",
                generation_config["top_k"]
            )
        )

        max_new_tokens = max(
            1,
            min(max_new_tokens, 100)
        )

        top_k = max(
            1,
            min(top_k, 50)
        )

        temperature = max(
            temperature,
            0.01
        )

        token_ids = _tokenizer.encode(prompt)

        idx = torch.tensor(
            [token_ids],
            dtype=torch.long
        )

        with torch.inference_mode():
            output = _model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        text = _tokenizer.decode(
            output[0].tolist()
        )

        return jsonify({
            "text": text
        })

    except Exception as exc:
        app.logger.exception(
            "Generation failed"
        )

        return jsonify({
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
