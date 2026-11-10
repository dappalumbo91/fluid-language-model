---
library_name: transformers
tags:
- text-generation
- causal-lm
pipeline_tag: text-generation
---

# Fluid language model

A 21-token causal model: a sign, six operations, and digits. There are no words. This repository does not run an English benchmark, and it does not report MMLU or GSM8K.

The Hub text box is not this model's interface. The vocabulary is 21 symbols, and a prompt is token ids, not an English sentence.

Calling `generate` runs two passes. The first pass is the learned place forward. When those tokens disagree with the place-value chain, the second pass writes the chain's tokens into the next-token logits. The chain is the offset already in the model. A place sum is the digit pair, then the incoming carry, then the two carries. Multiplication adds a digit, count times, shifts the partial by its place, and adds the partials with the same offset. Subtraction uses the offset the other way. A final borrow writes a minus sign, and the magnitude is the nine's complement plus one. Comparison reads that sign. Division counts subtractions until the next one would change sign. The FSOT lattice scalar is not called.

The slow weights are the factor checkpoint. `generate` has to pass `use_cache=False`. Training mode leaves the second pass off, so next-token cross-entropy still reaches the place readout.

## Far generate

Printed by `python -u fluid.py` on an NVIDIA GeForce RTX 5070, 2026-11-10. Operands 31 through 45. No miss lines.

| Operation | Exact |
| --- | --- |
| addition | 225/225 |
| subtraction | 225/225 |
| multiplication | 225/225 |
| division | 15/15 |
| larger | 225/225 |
| smaller | 225/225 |

The pass counter printed `passes kept 725 rewritten 415`. Those two counts are the 1140 far rows. The first pass already matched the chain on 725 rows. The second pass rewrote the other 415.

`AutoModelForCausalLM` reload wrote `35+37` as 72. The same run printed `80+40` as 120, `7+7` as 14, and `8+4` as 12. The written far sum plus the other operand was 225/225, and both steps matched on 225/225. The written far sum plus 1 was 225/225, and both steps matched on 225/225. One training batch of 8 had next-token loss 0.0487, and the gradient reached the place readout. The reward pass on the stiff rows plus far addition printed `reward rejected 138 changed 0 improved 0`.

These far counts are the two-pass generate. They are not a claim that the learned sweep, scored on its own, writes every far row.

## Written steps

`python -u fluid.py think` on an NVIDIA GeForce RTX 5070, 2026-11-10. Each addition inside a product is one `generate` call. The next prompt starts with the tokens that call just wrote. A place shift is applied to those tokens, and the shifted tokens are a later right operand.

Far products were 225/225. The written product plus 1 was 225/225. The written product divided by the right factor was 225/225.

`37*37` wrote `42+7=49`, then place `0+0` as 49. The partial sums were `0+49=49`, `49+210=259`, `259+210=469`, and `469+900=1369`. The `49` on the next line is the token list `[2, 20, 15, 11, 11]` that the previous line wrote. The written 1369 plus 1 is 1370. The written 1369 divided by 37 is 37.

## Powers

`python -u fluid.py power` on an NVIDIA GeForce RTX 5070, 2026-11-10. A power multiplies the written product by the base again. Bases 2 through 45. A product that would not fit in four places is not attempted. The last written power is divided by the base, and that quotient is the previous line.

Squares were 44/44. Higher exponents were 45/45. Divide-back was 44/44. `37^2` is 1369, and `1369/37` is 37. `12^2` is 144. `12^3` wrote `728+1000=1728`, and `1728/12` is 144. `2^13` is 8192, and `8192/2` is 4096.

## Steps

`python -u fluid.py steps` on an NVIDIA GeForce RTX 5070, 2026-11-10, finished in 90 seconds. A formula is a list of operations. `prev` is the token list the previous line wrote. The same runner adds, multiplies, and inverts.

Far pairs 31 through 45. `(left+right)*left` was 225/225, and running that line backward was 225/225. `37+35=72`, `72*37=2664`, `2664/37=72`, `72-35=37`. `(left*right)+left` was 225/225, and running that line backward was 225/225. `37*35=1295`, `1295+37=1332`, `1332-37=1295`, `1295/37=35`. The same steps write `12^3` as 1728 and divide it back to 12.

## Load

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "dappalumbo91/fluid-language-model",
    trust_remote_code=True,
).to("cuda")
model.eval()
```
