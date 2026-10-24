# Colab Talk — Digits on Demand

Script for a slide-free talk: only the Colab notebook on screen; the slide deck is attached separately. Plain language — every technical term gets a one-line everyday explanation. Timing ≈ 6:25 plus questions. The through-line for the three image sets: **breadth → precision → control** (all classes → one class ×10 → the guidance knob). All numbers are measured: 228,897 parameters · 30 epochs · 348 s · loss 1.105 → 0.040 · 10 digits ≈ 5 s on a T4 GPU (≈ 20 s on CPU).

**Legend:** plain text — read aloud; *(italics in parentheses — actions)*; `code` — what is on screen.

---

## 1. Opening (0:30)

*(The notebook is open, scrolled to the top. Preparation is already done: the repo is cloned, the sampler warmed up.)*

"Hi, I'm Aleh. My project is called Digits on Demand. It's a small AI model that draws handwritten digits — and it draws the digit you ask for. You say 'seven', it draws a seven. Not from memory — it draws a new seven every time, one that never existed before.

Everything you'll see is in this one notebook, and I'll run it live."

---

## 2. How the idea was born (0:35)

"A quick story of how I got here. The project rules were short: the code must run — live. And from the lectures we knew the grading logic: download someone else's model and show it — that's a B; build something yourself — you can aim higher.

I looked at four ideas: fine-tuning a vision transformer, training a small text transformer on my own data, image segmentation, and this one. One by one, the first three lost to a simple question: what can I build end to end, understand completely, and run live without fear? And the answer had actually been suggested by our professor in class: 'train a toy diffusion model and ask it to write a three.' Diffusion is also the biggest single topic on our final exam. So the choice made itself: a small model, built by me, with a demo the audience controls."

**Stage note:** all of this is factual: four candidates were really compared (LoRA-ViT on plant disease, a char-level transformer, U-Net segmentation, diffusion); the professor's line is from the 10/07 lecture, and so is the download-is-a-B grading logic. If asked about the rejected options: ViT-LoRA — "shows adaptation, not mechanism"; segmentation — the heaviest training of the four; text — not visual enough for a live audience.

---

## 3. The big idea (1:00)

"First, thirty seconds on how it works, in plain words.

The model is a diffusion model. Think of a drop of milk stirred into coffee — easy to mix, impossible to un-mix. Here, the 'mixing' is adding noise to a picture, step by step, until nothing is left but static — like an old broken TV screen. That part is easy and needs no intelligence.

The hard part is going backwards: starting from pure static and removing the noise, step by step, until a digit appears. That's what the network learns. It learned it from MNIST — sixty thousand handwritten digits, a classic beginner-friendly dataset.

And one extra trick, which is the whole point of my project: at every cleaning step, I also tell the network *which* digit I want. So it doesn't draw *a* digit — it draws *the* digit you ordered. That's called conditioning, and it's the same idea that lets Stable Diffusion follow a text prompt. Mine is just a much smaller version — small on purpose, so one person can build, train, and explain the whole thing."

**Stage note:** this is the longest monologue; everything after it is shorter and livelier.

---

## 4. What's on screen (0:55)

*(Point the cursor at the first cell — the clone.)*

"This notebook pulls the whole project from my GitHub — one Python file, about four hundred lines, plus a trained model. I trained it myself this week: thirty epochs, three hundred forty-eight seconds — under six minutes — on a free Google Colab GPU. The model is small: about two hundred thirty thousand parameters. Stable Diffusion is thousands of times bigger — but the recipe inside is the same.

*(Scroll to the `samples.png` grid below the training log.)*

And right under the training log — the model's own final exam. The moment training ends, the script asks the model to draw two of every digit, zero through nine. Every column is a requested class. So before the live part even starts, you can see that ordering works across the board.

To be upfront about the method: the dataset and one reference repo I downloaded, as a baseline. The code is AI-assisted — read, verified, and edited by me, and every design decision is mine to defend. That's the workflow this course taught us."

**Stage note:** the AI-assisted formula is mandatory — do not cut it if short on time.

---

## 5. A digit from the audience (0:20 + sampling time)

"Now the fun part. Someone give me a digit, zero through nine."

*(Say the audience names 7. Click into the `--sample-only --label 3` cell, change one digit: `3` → `7`. Shift+Enter with the cursor in that cell.)*

"Running. While it works, here's what's happening: the model starts from pure static and calls the network four hundred times. Each call asks one simple question: 'how much noise is in this picture right now?' — and subtracts the answer. And at every single call, the label 'seven' rides along, like a passenger telling the driver where to go."

*(If running on CPU, add:)* "On the GPU these four hundred calls take five seconds; the CPU does them one by one — about twenty seconds."

---

## 6. The reveal (0:30)

*(The `Image('demo.png')` cell → Shift+Enter. Ten digits appear. Hold a 2–3 second silent pause.)*

"There's your seven. The grid you saw earlier was two of every class — breadth. This is the opposite: one class, ten takes. And look, they're all different — different slant, different thickness, different handwriting. That's because each one started from its own random static. The model didn't memorize a seven; it learned what 'seven-ness' looks like."

**Stage note:** this is the climax — don't rush it; let the audience look.

---

## 7. The guidance knob (0:45 + sampling time)

"One more thing — the model has a volume knob.

During training, ten percent of the time I hid the label from the network. So it learned two skills: drawing the digit you ordered, and just drawing *something*. At generation time I can mix those two skills, and the mix ratio is a knob I control."

*(Click into the cell with the two guidance lines, change both `--label 3` to the audience's digit. Shift+Enter.)*

"I'm running the same seven twice: knob at zero, and knob at four."

*(The display cell → Shift+Enter. Two grids.)*

"Same order, same model — the only thing that changed between these two grids is the knob. At zero — more freedom, more variety, but some are barely readable. At four — cleaner, textbook sevens, but they all look alike. Variety versus legibility — one knob. Stable Diffusion gives its users exactly the same trade-off."

**Stage note:** if the top grid has a digit that fell apart, point at it: "this one barely made it — that's what low guidance costs you."

---

## 8. Close (0:30)

"To wrap up. The entire recipe of modern image generators — noise it, learn to un-noise it, steer it with a label — fits in one file and runs on a free GPU in six minutes. Making it draw *what you ask* took exactly one extra ingredient: an embedding for the label.

Everything is public: code, the trained model, and the samples are on my GitHub, together with the slides. Thank you — happy to take questions."

---

## Q&A — short answers

- **Why so small? Why not fine-tune Stable Diffusion?** "Building it myself means I understand every line. Fine-tuning shows adaptation; this shows the mechanism. And it trains in six minutes, so I can afford to break it and retrain."
- **Why does it take five seconds?** "Four hundred network calls, one per cleaning step. Production models cut that to thirty or fifty steps with smarter samplers — that's a natural next step."
- **Did AI write the code?** "Yes — the way this course taught us to use it. I framed the problem, chose the architecture, read and edited the code, and verified every number. Ask me why any piece is there."
- **What's the loss, and why does it start at 1.0?** "The network predicts the noise, and the loss is mean squared error. An untrained network guesses zero, and the average squared size of standard noise is exactly one — so training starts at 1.0 by definition. Mine ended at 0.04."
- **Next step?** "Swap the U-Net for a small transformer — the DiT architecture — and compare at the same size."

---

## Action checklist (demo order)

1. **Before the talk, silently:** open the notebook → check the runtime → run the clone cell → one warm-up `--sample-only` run (don't show the result).
2. Blocks 1–4 — talk while scrolling toward the generation cell.
3. Audience digit → **1 edit** (`--label`) → Shift+Enter (cursor inside the cell!).
4. The `demo.png` display cell → Shift+Enter → pause.
5. The guidance cell → **2 edits** (both `--label`) → Shift+Enter (~10 s on GPU, ~40 s on CPU; the output must show **two** `saved` blocks).
6. The g0/g4 display cell → Shift+Enter.
7. Close. If something fails — re-run the same cell; full recovery from scratch = clone (~1 minute; the checkpoint comes from the repo).
8. Image captions, if lost: `samples.png` — "two of every class, the model's own final exam" (breadth); `demo.png` — "one class, ten takes" (precision); `g0`/`g4` — "same order, two settings of the knob" (control).
