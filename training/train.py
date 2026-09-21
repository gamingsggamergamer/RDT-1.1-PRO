import json
from pathlib import Path
import torch

from tokenizer.tokenizer import ByteBPETokenizer
from model.roblox_llm import RobloxLLM
from training.dataset import load_dataset, build_tokenizer, make_splits, get_batch

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "config.json").read_text())

data_path = ROOT / "data" / "roblox_level2_dataset.jsonl"
tok_path = ROOT / "tokenizer" / "roblox_bpe_tokenizer.json"
ckpt_path = ROOT / "checkpoints" / "roblox_level2_llm.pt"

text = load_dataset(data_path)
if tok_path.exists():
    tokenizer = ByteBPETokenizer.load(tok_path)
else:
    tokenizer = build_tokenizer(text, tok_path, cfg["tokenizer"]["max_merges"])

ids = tokenizer.encode(text)
train_data, val_data = make_splits(ids, cfg["model"]["block_size"], cfg["training"]["val_fraction"])

device = "cuda" if torch.cuda.is_available() else "cpu"
m = cfg["model"]
model = RobloxLLM(
    vocab_size=tokenizer.vocab_size,
    n_embd=m["n_embd"],
    n_heads=m["n_heads"],
    n_layers=m["n_layers"],
    block_size=m["block_size"],
    dropout=m["dropout"]
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"]
)

model.train()
for step in range(cfg["training"]["max_steps"]):
    xb, yb = get_batch(train_data, m["block_size"], cfg["training"]["batch_size"], device)
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
    optimizer.step()

    if step % cfg["training"]["eval_interval"] == 0:
        print(f"step={step} loss={loss.item():.4f}")

ckpt_path.parent.mkdir(exist_ok=True)
torch.save({
    "model": model.state_dict(),
    "vocab_size": tokenizer.vocab_size,
    "config": m
}, ckpt_path)
print(f"Saved checkpoint: {ckpt_path}")
