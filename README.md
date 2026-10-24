# Digits on Demand: Conditional Diffusion on MNIST

**Course:** CSE/MB620 — Deep Learning with PyTorch (CSTU, Fall 2026)
**Author:** Aleh Haiko
**Final project** — presented October 24, 2026

## Project overview

A class-conditional DDPM built from scratch in PyTorch. Plain diffusion generates *an* image; this model generates *the* image you ask for: `--label 3` draws a 3 out of pure noise. The same conditioning idea that lets Stable Diffusion follow a text prompt, scaled down to ten digit classes so the whole thing trains end to end in about six minutes on a free Colab GPU.

**Downloaded:** the MNIST dataset, and [TeaPearce/Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST) as a comparison baseline only.
**Built:** the denoiser, the conditioning, the guidance, and the training/sampling loops — all in one file. The code is AI-assisted (Claude) — read, verified, and edited by me; every design decision and every reported number is mine to defend.

## Repository contents

| File | What it is |
|---|---|
| `mnist_conditional_diffusion.py` | The whole project: model, training, sampling, CLI |
| `samples_smoke_test.png` | Offline smoke-test output (sklearn digits, 16×16, 132 s on CPU) |
| `samples_mnist.png` | Generated digits, one row per requested class (full MNIST run) |
| `cond_diffusion.pt` | Trained checkpoint (full MNIST run) |
| `requirements.txt` | torch, torchvision, matplotlib (+ scikit-learn for the offline test) |

## Problem and data

Conditional image generation: control which digit the model draws, not just whether it draws a digit. Data: MNIST, 60,000 labeled 28×28 grayscale digits; the labels are free and become the condition, not a classification target. For an offline smoke test (no network needed) the script can run on sklearn's built-in 8×8 digits instead.

## Solution

| Component | Choice |
|---|---|
| Diffusion | DDPM, linear beta schedule 1e-4 → 0.02, T = 400 steps |
| Denoiser | Compact U-Net: 2 downs, bottleneck, 2 ups with concatenation skips |
| Timestep | Sinusoidal embedding, injected into every block |
| Condition | `nn.Embedding(11, d)` — 10 digit labels + a learned "no label" token |
| Guidance | Classifier-free: labels dropped 10% in training; at sampling `eps = (1+w)·cond − w·uncond` |
| Loss | MSE between true and predicted noise, random timestep per image |
| Size | 228,897 parameters (~0.23M) |

## Results

| Metric | Value |
|---|---|
| Parameters | 228,897 (~0.23M) |
| Training | 30 epochs, 348 s (~6 min), Colab T4 |
| Final loss (MSE on noise) | 1.105 -> 0.040 |
| Sampling 10 digits | ~5 s on T4 |

Offline smoke test (committed as `samples_smoke_test.png`): sklearn digits, 62k parameters, T = 200, 120 epochs, 132 s on CPU — loss 1.08 → 0.098, all ten requested classes legible.

## Run the live demo

**Google Colab (GPU):**

```bash
pip install -r requirements.txt
# train (~20 min on T4), saves cond_diffusion.pt:
python mnist_conditional_diffusion.py --dataset mnist --epochs 30
# then sample any digit:
python mnist_conditional_diffusion.py --sample-only --label 3 --out demo.png
```

**Offline / CPU (no network needed):**

```bash
python mnist_conditional_diffusion.py --dataset digits --img-size 16 --timesteps 200 --epochs 120 --base 16
```

**The guidance knob:** `--guidance 0` gives more variety, `--guidance 4` gives cleaner, more stereotyped digits — same seed, visibly different outputs.

## Notes and limitations

- MNIST is a toy domain by design ("think small"): the point is the full pipeline, not state of the art.
- The demo digits are judged by a separate classifier, not by eye; the live demo sample is not a measure of overall accuracy.
- Next step: swap the U-Net denoiser for a small transformer (the DiT move) and compare at equal parameter count.
