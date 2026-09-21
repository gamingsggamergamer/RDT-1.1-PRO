# Syncing your real Colab artifacts

GitHub and Render cannot silently read files from your Google Colab runtime.

To preserve your exact current Level 2 dataset/tokenizer:

1. In Colab, save/download:
   - `roblox_level2_dataset.jsonl`
   - `roblox_bpe_tokenizer.json`
2. Put them into this repository at:
   - `data/roblox_level2_dataset.jsonl`
   - `tokenizer/roblox_bpe_tokenizer.json`
3. Commit/push to GitHub.
4. Render will deploy the GitHub repository.

The ZIP includes a small seed dataset so the repository has a working structure immediately.
