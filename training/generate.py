import json
from pathlib import Path
import torch

from tokenizer.tokenizer import ByteBPETokenizer
from model.roblox_llm import RobloxLLM

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "config.json").read_text())
tok = ByteBPETokenizer.load(ROOT / "tokenizer" / "roblox_bpe_tokenizer.json")
ckpt = torch.load(ROOT / "checkpoints" / "roblox_level2_llm.pt", map_location="cpu")

m = ckpt["config"]
model = RobloxLLM(
    vocab_size=ckpt["vocab_size"],
    n_embd=m["n_embd"],
    n_heads=m["n_heads"],
    n_layers=m["n_layers"],
    block_size=m["block_size"],
    dropout=m["dropout"]
)
model.load_state_dict(ckpt["model"])
model.eval()

prompt = "local Players = game:GetService(\"Players\")"
idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
out = model.generate(
    idx,
    max_new_tokens=cfg["generation"]["max_new_tokens"],
    temperature=cfg["generation"]["temperature"],
    top_k=cfg["generation"]["top_k"]
)
print(tok.decode(out[0].tolist()))
