# Colab Talk — Digits on Demand (вся презентация из ноутбука)

Сценарий без слайдов: на экране только Colab, PPTX приложен в Classroom. Язык простой, каждый термин сразу объясняется бытовым отступлением. Хронометраж ~6:25 + вопросы. Логика трёх картинок, которую держишь в голове: **breadth → precision → control** (все классы → один класс ×10 → ручка guidance). Числа — измеренные: 228,897 параметров · 30 эпох · 348 с · лосс 1.105 → 0.040 · 10 цифр ≈ 5 с.

**Легенда:** обычный текст — читаешь вслух; *курсив в скобках* — действия; `код` — что на экране.

---

## 1. Открытие (0:30)

*(Открыт ноутбук, прокручен к началу. Подготовка уже сделана: clone выполнен, сэмплер прогрет.)*

"Hi, I'm Aleh. My project is called Digits on Demand. It's a small AI model that draws handwritten digits — and it draws the digit you ask for. You say 'seven', it draws a seven. Not from memory — it draws a new seven every time, one that never existed before.

Everything you'll see is in this one notebook, and I'll run it live."

---

## 2. Как родилась идея (0:35)

"A quick story of how I got here. The project rules were short: the code must run — live. And from the lectures we knew the grading logic: download someone else's model and show it — that's a B; build something yourself — you can aim higher.

I looked at four ideas: fine-tuning a vision transformer, training a small text transformer on my own data, image segmentation, and this one. One by one, the first three lost to a simple question: what can I build end to end, understand completely, and run live without fear? And the answer had actually been suggested by our professor in class: 'train a toy diffusion model and ask it to write a three.' Diffusion is also the biggest single topic on our final exam. So the choice made itself: a small model, built by me, with a demo the audience controls."

**Ремарка:** всё здесь — факты: четыре кандидата реально сравнивались (LoRA-ViT на болезнях растений, char-трансформер, U-Net-сегментация, диффузия); фраза профессора — с лекции 10/07; логика оценок «download = B, build = выше» — оттуда же. Если спросят про отвергнутые варианты: ViT-LoRA — «shows adaptation, not mechanism», сегментация — самое тяжёлое обучение из четырёх, текст — не визуален для зала.

---

## 3. Большая идея (1:00)

"First, thirty seconds on how it works, in plain words.

The model is a diffusion model. Think of a drop of milk stirred into coffee — easy to mix, impossible to un-mix. Here, the 'mixing' is adding noise to a picture, step by step, until nothing is left but static — like an old broken TV screen. That part is easy and needs no intelligence.

The hard part is going backwards: starting from pure static and removing the noise, step by step, until a digit appears. That's what the network learns. It learned it from MNIST — sixty thousand handwritten digits, a classic beginner-friendly dataset.

And one extra trick, which is the whole point of my project: at every cleaning step, I also tell the network *which* digit I want. So it doesn't draw *a* digit — it draws *the* digit you ordered. That's called conditioning, and it's the same idea that lets Stable Diffusion follow a text prompt. Mine is just a much smaller version — small on purpose, so one person can build, train, and explain the whole thing."

**Ремарка:** это самый длинный монолог; дальше всё короче и живее.

---

## 4. Что на экране (0:55)

*(Показываешь курсором на первую ячейку — clone.)*

"This notebook pulls the whole project from my GitHub — one Python file, about four hundred lines, plus a trained model. I trained it myself this week: thirty epochs, three hundred forty-eight seconds — under six minutes — on a free Google Colab GPU. The model is small: about two hundred thirty thousand parameters. Stable Diffusion is thousands of times bigger — but the recipe inside is the same.

*(Прокручиваешь к сетке `samples.png` под логом тренировки.)*

And right under the training log — the model's own final exam. The moment training ends, the script asks the model to draw two of every digit, zero through nine. Every column is a requested class. So before the live part even starts, you can see that ordering works across the board.

To be upfront about the method: the dataset and one reference repo I downloaded, as a baseline. The code is AI-assisted — read, verified, and edited by me, and every design decision is mine to defend. That's the workflow this course taught us."

**Ремарка:** формула AI-assisted — обязательная, не выкидывать при нехватке времени.

---

## 5. Цифра из зала (0:20 + 5 с генерации)

"Now the fun part. Someone give me a digit, zero through nine."

*(Называют, допустим, 7. Клик в ячейку `--sample-only --label 3`, меняешь одну цифру: `3` → `7`. Shift+Enter, курсор в этой ячейке.)*

"Running. While it works — about five seconds — here's what's happening: the model starts from pure static and calls the network four hundred times. Each call asks one simple question: 'how much noise is in this picture right now?' — and subtracts the answer. And at every single call, the label 'seven' rides along, like a passenger telling the driver where to go."

---

## 6. Показ (0:30)

*(Ячейка `Image('demo.png')` → Shift+Enter. Появляются десять цифр. Пауза 2–3 секунды — молча.)*

"There's your seven. The grid you saw earlier was two of every class — breadth. This is the opposite: one class, ten takes. And look, they're all different — different slant, different thickness, different handwriting. That's because each one started from its own random static. The model didn't memorize a seven; it learned what 'seven-ness' looks like."

**Ремарка:** это кульминация — не торопись, дай залу рассмотреть.

---

## 7. Ручка guidance (0:45 + 10 с генерации)

"One more thing — the model has a volume knob.

During training, ten percent of the time I hid the label from the network. So it learned two skills: drawing the digit you ordered, and just drawing *something*. At generation time I can mix those two skills, and the mix ratio is a knob I control."

*(Клик в ячейку с двумя строками guidance, меняешь обе `--label 3` на цифру зала. Shift+Enter. ~10 секунд.)*

"I'm running the same seven twice: knob at zero, and knob at four."

*(Ячейка показа → Shift+Enter. Две сетки.)*

"Same order, same model — the only thing that changed between these two grids is the knob. At zero — more freedom, more variety, but some are barely readable. At four — cleaner, textbook sevens, but they all look alike. Variety versus legibility — one knob. Stable Diffusion gives its users exactly the same trade-off."

**Ремарка:** если в верхней сетке попалась совсем кривая цифра — ткни в неё: "this one barely made it — that's what low guidance costs you."

---

## 8. Финал (0:30)

"To wrap up. The entire recipe of modern image generators — noise it, learn to un-noise it, steer it with a label — fits in one file and runs on a free GPU in six minutes. Making it draw *what you ask* took exactly one extra ingredient: an embedding for the label.

Everything is public: code, the trained model, and the samples are on my GitHub, and the link is in Classroom, together with the slides. Thank you — happy to take questions."

---

## Q&A — короткие ответы (EN)

- **Why so small? Why not fine-tune Stable Diffusion?** "Building it myself means I understand every line. Fine-tuning shows adaptation; this shows the mechanism. And it trains in six minutes, so I can afford to break it and retrain."
- **Why does it take five seconds?** "Four hundred network calls, one per cleaning step. Production models cut that to thirty or fifty steps with smarter samplers — that's a natural next step."
- **Did AI write the code?** "Yes — the way this course taught us to use it. I framed the problem, chose the architecture, read and edited the code, and verified every number. Ask me why any piece is there."
- **What's the loss, and why does it start at 1.0?** "The network predicts the noise, and the loss is mean squared error. An untrained network guesses zero, and the average squared size of standard noise is exactly one — so training starts at 1.0 by definition. Mine ended at 0.04."
- **Next step?** "Swap the U-Net for a small transformer — the DiT architecture — and compare at the same size."

---

## Шпаргалка действий (порядок на демо)

1. **До выступления, молча:** открыть ноутбук → проверить T4 → ячейка clone → прогрев `--sample-only` один раз (результат не показывать).
2. Блоки 1–4 — говоришь, прокручивая к ячейке генерации.
3. Цифра из зала → **1 правка** (`--label`) → Shift+Enter (курсор в ячейке!).
4. Ячейка показа `demo.png` → Shift+Enter → пауза.
5. Ячейка guidance → **2 правки** (обе `--label`) → Shift+Enter (~10 с, в выводе должно быть **два** блока `saved`).
6. Ячейка показа g0/g4 → Shift+Enter.
7. Финал. Если что-то упало — перезапустить ту же ячейку; полное восстановление с нуля = clone (~1 минута, чекпойнт приедет из репозитория).
8. Подписи к картинкам, если собьёшься: `samples.png` — «two of every class, the model's own final exam» (широта); `demo.png` — «one class, ten takes» (точность); `g0`/`g4` — «same order, two settings of the knob» (контроль).
