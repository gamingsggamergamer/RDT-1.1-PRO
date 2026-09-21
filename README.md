# Roblox LLM — Level 2

A from-scratch Roblox/Luau language model project using PyTorch.

## Run locally

```bash
pip install -r requirements.txt
python training/train.py
python training/generate.py
```

## Render

The Flask API starts with:

```bash
gunicorn api.server:app --bind 0.0.0.0:$PORT
```

`render.yaml` is included for deployment.

## Important

The repository contains the Level 2 seed dataset structure. The exact files from a private Google Colab runtime cannot be pulled automatically into this ZIP. Export your current Colab dataset/tokenizer into `data/` and `tokenizer/` if you want to preserve those exact artifacts.
