import json
import torch
from pathlib import Path
from tokenizer.tokenizer import ByteBPETokenizer

def load_dataset(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return "\n\n".join(row["content"] for row in rows)

def build_tokenizer(text, path, max_merges=300):
    tokens = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    next_id = 256

    for _ in range(max_merges):
        if len(tokens) < 2:
            break
        counts = {}
        for pair in zip(tokens[:-1], tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        pair, _ = max(counts.items(), key=lambda x: x[1])
        new_bytes = vocab[pair[0]] + vocab[pair[1]]
        vocab[next_id] = new_bytes
        merges.append((pair[0], pair[1], next_id))

        out, i = [], 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                out.append(next_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
        next_id += 1

    tok = ByteBPETokenizer(vocab, merges)
    tok.save(path)
    return tok

def make_splits(token_ids, block_size, val_fraction=0.1):
    data = torch.tensor(token_ids, dtype=torch.long)
    cut = int(len(data) * (1 - val_fraction))
    return data[:cut], data[cut:]

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
