import json
from pathlib import Path

class ByteBPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = {int(k): bytes(v) for k, v in (vocab or {}).items()}
        self.merges = [tuple(x) for x in (merges or [])]
        self.merge_lookup = {(a, b): c for a, b, c in self.merges}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            pairs = list(zip(tokens[:-1], tokens[1:]))
            valid = [p for p in pairs if p in self.merge_lookup]
            if not valid:
                break
            pair = min(valid, key=lambda p: self.merge_lookup[p])
            new_token = self.merge_lookup[pair]
            out, i = [], 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                    out.append(new_token)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def decode(self, tokens):
        raw = b"".join(self.vocab[t] for t in tokens)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": [list(x) for x in self.merges]
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], data["merges"])
