
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, request, jsonify
from flask_cors import CORS


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHECKPOINT = os.path.join(
    ROOT,
    "checkpoints",
    "level4",
    "rdt_1_1_pro_level4.pt"
)

TOKENIZER_PATH = os.path.join(
    ROOT,
    "tokenizer",
    "level4_tokenizer.json"
)


# ============================================================
# MODEL
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head
    ):

        super().__init__()

        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd
        )

        self.out_proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    2048,
                    2048
                )
            ).view(
                1,
                1,
                2048,
                2048
            ),
            persistent=False
        )

    def forward(self, x):

        b, t, c = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        q = q.view(
            b,
            t,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            b,
            t,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            b,
            t,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        attention = (
            q @ k.transpose(-2, -1)
        ) / (
            self.head_dim ** 0.5
        )

        attention = attention.masked_fill(
            self.mask[
                :,
                :,
                :t,
                :t
            ] == 0,
            float("-inf")
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

        output = attention @ v

        output = output.transpose(
            1,
            2
        ).contiguous().view(
            b,
            t,
            c
        )

        return self.out_proj(
            output
        )


class FeedForward(nn.Module):

    def __init__(
        self,
        n_embd,
        ffn_dim
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            n_embd,
            ffn_dim
        )

        self.fc2 = nn.Linear(
            ffn_dim,
            n_embd
        )

        self.activation = nn.GELU()

    def forward(self, x):

        return self.fc2(
            self.activation(
                self.fc1(x)
            )
        )


class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        ffn_dim
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.attention = CausalSelfAttention(
            n_embd,
            n_head
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )

        self.ffn = FeedForward(
            n_embd,
            ffn_dim
        )

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


class RDT1BModel(nn.Module):

    def __init__(self):

        super().__init__()

        vocab_size = 32000
        context_length = 2048
        n_embd = 2048
        n_layer = 19
        n_head = 16
        ffn_dim = 8192

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            context_length,
            n_embd
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd,
                n_head,
                ffn_dim
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def forward(self, input_ids):

        b, t = input_ids.shape

        positions = torch.arange(
            t,
            device=input_ids.device
        )

        x = (
            self.token_embedding(
                input_ids
            )
            +
            self.position_embedding(
                positions
            )
        )

        for block in self.blocks:

            x = block(x)

        x = self.ln_f(x)

        return self.lm_head(x)


# ============================================================
# TOKENIZER
# ============================================================

try:

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    def encode_rdt(text):

        return tokenizer.encode(
            text
        ).ids

    def decode_rdt(ids):

        return tokenizer.decode(
            ids,
            skip_special_tokens=False
        )

except Exception as error:

    raise RuntimeError(
        "Tokenizer failed to load: "
        + str(error)
    )


PAD_ID = tokenizer.token_to_id(
    "<|pad|>"
)

END_ID = tokenizer.token_to_id(
    "<|end|>"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = RDT1BModel()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=20
):

    ids = encode_rdt(
        prompt
    )

    if not ids:
        ids = [PAD_ID]

    ids = ids[
        -2047:
    ]

    tokens = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(
        max_new_tokens
    ):

        context = tokens[
            :,
            -2048:
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda"
        ):

            logits = model(
                context
            )

        logits = logits[
            :,
            -1,
            :
        ].float()

        logits = logits / max(
            temperature,
            1e-5
        )

        values, indices = torch.topk(
            logits,
            min(
                top_k,
                logits.shape[-1]
            )
        )

        filtered = torch.full_like(
            logits,
            float("-inf")
        )

        filtered.scatter_(
            1,
            indices,
            values
        )

        probabilities = torch.softmax(
            filtered,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            1
        )

        tokens = torch.cat(
            [
                tokens,
                next_token
            ],
            dim=1
        )

        if (
            END_ID is not None
            and int(next_token.item()) == END_ID
        ):
            break

    return decode_rdt(
        tokens[0].tolist()
    )


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__
)

CORS(app)


@app.get("/")
def home():

    return jsonify({
        "name": "RDT-1.1 Pro",
        "status": "online",
        "model": "1.026B",
        "version": "Level 4"
    })


@app.get("/health")
def health():

    return jsonify({
        "status": "healthy",
        "model": "RDT-1.1 Pro",
        "parameters": 1026541568
    })


@app.post("/generate")
def generate_endpoint():

    data = request.get_json(
        silent=True
    ) or {}

    prompt = data.get(
        "prompt",
        ""
    )

    if not isinstance(
        prompt,
        str
    ):

        return jsonify({
            "error": "prompt must be a string"
        }), 400

    if not prompt.strip():

        return jsonify({
            "error": "prompt is empty"
        }), 400

    max_new_tokens = int(
        data.get(
            "max_new_tokens",
            80
        )
    )

    max_new_tokens = max(
        1,
        min(
            max_new_tokens,
            100
        )
    )

    temperature = float(
        data.get(
            "temperature",
            0.8
        )
    )

    top_k = int(
        data.get(
            "top_k",
            20
        )
    )

    try:

        output = generate(
            prompt,
            max_new_tokens,
            temperature,
            top_k
        )

        return jsonify({
            "model": "RDT-1.1 Pro",
            "response": output
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
