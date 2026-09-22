
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

HF_REPO = "mrfirex79/RDT-1.1-Pro"
MODEL_FILENAME = "rdt_1_1_pro_level4.pt"

CONFIG_PATH = ROOT / "rdt_1b_config.json"
TOKENIZER_PATH = ROOT / "tokenizer" / "level4_tokenizer.json"

DEVICE = torch.device("cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================
# MODEL
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, use_bias=True):
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=use_bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=use_bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=use_bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=use_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, n_embd, ffn_dim, use_bias=True):
        super().__init__()

        self.fc1 = nn.Linear(n_embd, ffn_dim, bias=use_bias)
        self.fc2 = nn.Linear(ffn_dim, n_embd, bias=use_bias)

        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class RDTTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        ffn_dim,
        dropout=0.0,
        use_bias=True,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.attention = CausalSelfAttention(
            n_embd,
            n_head,
            dropout,
            use_bias,
        )

        self.ln2 = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.feed_forward = FeedForward(
            n_embd,
            ffn_dim,
            use_bias,
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class RDT1BModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        architecture = config["architecture"]

        vocab_size = architecture["vocab_size"]
        context_length = architecture["context_length"]
        n_embd = architecture["n_embd"]
        n_layer = architecture["n_layer"]
        n_head = architecture["n_head"]
        ffn_dim = architecture["ffn_dim"]
        dropout = architecture.get("dropout", 0.0)
        use_bias = architecture.get("use_bias", True)
        tie_embeddings = architecture.get("tie_embeddings", True)

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_embd = n_embd

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd,
        )

        self.position_embedding = nn.Embedding(
            context_length,
            n_embd,
        )

        self.blocks = nn.ModuleList(
            [
                RDTTransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_bias=use_bias,
                )
                for _ in range(n_layer)
            ]
        )

        self.final_ln = nn.LayerNorm(
            n_embd,
            elementwise_affine=True,
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        B, T = input_ids.shape

        if T > self.context_length:
            input_ids = input_ids[:, -self.context_length:]
            T = input_ids.shape[1]

        positions = torch.arange(
            T,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        return self.lm_head(x)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# ============================================================
# LOAD TOKENIZER
# ============================================================

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"Tokenizer not found: {TOKENIZER_PATH}"
    )

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))


# ============================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

print("Downloading/loading RDT-1.1 Pro checkpoint...")

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO,
    filename=MODEL_FILENAME,
    repo_type="model",
    token=HF_TOKEN,
)

print(f"Checkpoint: {MODEL_PATH}")


# ============================================================
# LOAD MODEL
# ============================================================

model = RDT1BModel(CONFIG)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    raise RuntimeError("Unsupported checkpoint format.")

model.load_state_dict(
    state_dict,
    strict=True,
)

del checkpoint
del state_dict

model.eval()
model.to(DEVICE)

print(
    f"RDT-1.1 Pro loaded: "
    f"{sum(p.numel() for p in model.parameters()):,} parameters"
)


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate_text(
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=20,
):
    encoded = TOKENIZER.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE,
    )

    max_new_tokens = min(
        int(max_new_tokens),
        80,
    )

    temperature = max(
        float(temperature),
        0.05,
    )

    for _ in range(max_new_tokens):

        idx = input_ids[
            :,
            -CONFIG["architecture"]["context_length"] :
        ]

        logits = model(idx)

        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            k = min(
                int(top_k),
                logits.shape[-1],
            )

            values, _ = torch.topk(
                logits,
                k,
            )

            cutoff = values[:, [-1]]

            logits = torch.where(
                logits < cutoff,
                torch.full_like(
                    logits,
                    float("-inf"),
                ),
                logits,
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    output_ids = input_ids[0].tolist()

    return TOKENIZER.decode(
        output_ids,
        skip_special_tokens=True,
    )


# ============================================================
# FLASK API
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    },
)


@app.get("/")
def root():
    return jsonify(
        {
            "name": "RDT-1.1 Pro",
            "version": "Level 4",
            "status": "online",
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "model": "RDT-1.1 Pro",
            "parameters": sum(
                p.numel()
                for p in model.parameters()
            ),
            "device": "cpu",
        }
    )


@app.post("/generate")
def generate():
    data = request.get_json(
        silent=True
    ) or {}

    prompt = str(
        data.get(
            "prompt",
            "",
        )
    ).strip()

    if not prompt:
        return jsonify(
            {
                "error": "Prompt is required."
            }
        ), 400

    max_new_tokens = data.get(
        "max_new_tokens",
        80,
    )

    temperature = data.get(
        "temperature",
        0.8,
    )

    top_k = data.get(
        "top_k",
        20,
    )

    try:
        result = generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        return jsonify(
            {
                "response": result,
                "model": "RDT-1.1 Pro",
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                7860,
            )
        ),
    )
