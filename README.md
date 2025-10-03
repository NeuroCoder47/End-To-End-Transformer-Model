# Mini Transformer NMT (EN→IT)

> A compact, readable Transformer for machine translation with a friendly training loop, simple tokenizers, and clean modular components.

---

## Highlights

- Encoder–decoder Transformer with multi-head attention, feed-forward blocks, and sinusoidal positions, implemented from scratch for clarity.
- WordLevel tokenizers trained per-language with special tokens [SOS]/[EOS]/[PAD] and simple whitespace pre-tokenization.
- Greedy decoding for validation, label smoothing in loss, Adam optimizer, TensorBoard logging, and checkpointing on improvement.
- Default EN→IT setup with configurable sequence length, batch size, and model width via a small config.

---

## Table of contents

- Getting started
- Data & tokenization
- Model architecture
- Training & evaluation
- Paper comparison
- Benchmark references
- WMT14 BLEU (for this repo)
- Roadmap
- Repo map

---

## Getting started

1) Install dependencies: PyTorch, datasets, tokenizers, tensorboard, tqdm.

2) Run training: `python train.py` to kick off training and validation with default config.

3) Watch logs: `tensorboard --logdir runs/tmodel` to follow loss curves and sample predictions.

Notes:
- Adjust language pair, sequence length, and training epochs in `config.py` as needed.
- Weights are saved on validation improvement using an epoch-stamped file name.

---

## Data & tokenization

- Loads a translation dataset through the Hugging Face datasets API and splits into train/validation inside the training script.
- Trains two WordLevel tokenizers (source and target) with special tokens and frequency thresholding, then saves `tokenizer_en.json` and `tokenizer_it.json`.
- The dataset class adds [SOS]/[EOS], pads/truncates to `seq_len`, creates padding masks, and returns a causal mask for autoregressive decoding.

---

## Model architecture

- Token embeddings are scaled by \( \sqrt{d_{\text{model}}} \) and combined with sinusoidal positional encodings as in the original Transformer.
- Multi-head attention uses scaled dot-product \( \mathrm{softmax}(QK^\top/\sqrt{d_k})V \) with masking for padding and causality.
- Pre-norm residual blocks (LayerNorm → sublayer → dropout → residual) improve stability in smaller models.
- Encoder stacks N blocks of self-attention + feed-forward; decoder stacks masked self-attention, cross-attention, and feed-forward.
- A final linear projection maps decoder states to target vocabulary logits, with sensible initialization.

---

## Training & evaluation

- Optimization: Adam with fixed learning rate \(1\times10^{-4}\) and label smoothing \( \epsilon=0.1 \), ignoring PAD in the loss.
- Validation: greedy decoding from [SOS] until [EOS] or `max_len`, printing SOURCE/TARGET/PREDICTED triplets.
- Logging and checkpoints: TensorBoard logs under `runs/tmodel`; weights saved to `weights` with a consistent basename and epoch index.

---

## Paper comparison

The table contrasts defaults in this repo with a commonly referenced “Transformer (base)” configuration.

| Aspect | This repo | Transformer (base) |
|---|---|---|
| Layers (N) | 4 encoder + 4 decoder by default | 6 encoder + 6 decoder |
| d_model | 256 | 512 |
| Heads (h) | 4 | 8 |
| d_ff | 1024 | 2048 |
| Norm placement | Pre‑norm | Post‑norm in original description |
| Positional encoding | Sinusoidal | Sinusoidal |
| Tokenization | WordLevel (per language) | Subword/BPE common in practice |
| LR schedule | Fixed \(1\times10^{-4}\) | Warmup + inverse‑sqrt (e.g., 4000 steps) |
| Label smoothing | 0.1 | 0.1 |
| Decoding | Greedy | Beam search for reported BLEU |

---

## Benchmark references

The original paper evaluated on WMT14 translation tasks and reported strong results with Transformer (base/big) configurations.

- WMT14 EN→DE: BLEU ≈ 28.4, representing a significant improvement over earlier NMT systems.
- WMT14 EN→FR: BLEU ≈ 41.8 in the reported setup with large compute and beam decoding.

Educational references summarize standard base hyperparameters: N=6, d_model=512, d_ff=2048, h=8, warmup scheduling, and label smoothing 0.1.

---

## WMT14 BLEU (for this repo)

These are realistic, clearly labeled expectations for this codebase if trained directly on WMT14 with current defaults, based on the capacity gap, absence of warmup scheduling, and greedy decoding versus beam search; actual scores require full training and evaluation to confirm.

- EN→DE: substantially below ≈23.74 BLEU due to smaller width/depth, fewer heads, lack of warmup, and greedy decoding; expect markedly lower BLEU without scaling model/training and adding beam search.
- EN→FR: below ≈40.18 BLEU for the same reasons; increasing parameters, training duration, and using a proper schedule/beam would be necessary to close the gap.

Tip:
- To approach paper-like scores, align hyperparameters, adopt warmup + inverse‑sqrt decay, train longer on WMT14, and decode with beam search.

---

## Roadmap

- Add warmup + inverse‑sqrt scheduler and gradient clipping.
- Switch to a subword tokenizer and add beam search decoding for evaluation.
- Integrate sacreBLEU/WER evaluation and proper WMT14 data loaders.

---

## Repo map

- train.py — data loading, tokenizer training, train/val loops, greedy decode, TensorBoard, checkpoints.
- dataset.py — BilingualDataset with special tokens, padding/causal masks, and item collation.
- model2.py — embeddings, sinusoidal positions, multi-head attention, pre‑norm blocks, encoder/decoder, projection.
- config.py — hyperparameters, model/tokenizer paths, experiment naming, and utility for weight paths.

---

## Screenshots & animations

- Place training curves under `assets/curves.png` and a short decode demo as `assets/demo.gif`, then reference them in the header and this section if desired.

---

## Citation pointers

- “Attention Is All You Need” — core design and WMT14 BLEU results.
- “The Annotated Transformer” — canonical base hyperparameters and schedule.
