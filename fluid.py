"""Causal language-model interface over the place forward.

Hugging Face calls this the way it calls any causal model: token ids in,
next-token logits out, and ``generate`` appends the tokens just written.
Answer digits still come from the place offset. A learned head does not
get to drop the carry.

Slow weights take ordinary next-token cross-entropy. The fast residual is
a synapse. Its trace is the hidden state that just fired. A failed domain
check releases reward -1, and a passed check would release +1. The update
is that reward times the trace, scaled to the existing fast step of 0.1.
The integer answer is not the target.

There are no words yet. This file does not run an English benchmark.

The first pass is the learned place forward. When its tokens disagree with
the place-value chain, a second pass writes the chain's tokens. The chain
is the offset already in the model. Weights load from ``weights/place/factor.pt``.

    python -u fluid.py
"""

from __future__ import annotations

import shutil

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedTokenizerFast
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

import place
import scratch

VOCAB = scratch.VOCAB
PROMPT_LEN = scratch.PROMPT_LEN
ANSWER_LEN = scratch.ANSWER_LEN
PLUS = scratch.PLUS_SIGN
MINUS = scratch.MINUS_SIGN
DIGIT0 = scratch.DIGIT0
EQ = scratch.EQ
OP_ADD = scratch.OP_ADD
OP_SMALLER = scratch.OP_SMALLER
FILL = -1.0e4
FAST_LR = 0.1
SYMBOLS = (
    ["<yes>", "<no>", "+", "-", "add", "sub", "mul", "div", "larger", "smaller", "="]
    + [str(digit) for digit in range(10)]
)


class FluidConfig(PretrainedConfig):
    model_type = "fluid"

    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = place.H, **kwargs) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("bos_token_id", PLUS)
        kwargs["use_cache"] = False
        super().__init__(**kwargs)
        self.auto_map = {
            "AutoConfig": "fluid.FluidConfig",
            "AutoModelForCausalLM": "fluid.FluidForCausalLM",
        }


class FluidForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FluidConfig
    base_model_prefix = "place"

    def __init__(self, config: FluidConfig) -> None:
        super().__init__(config)
        self.place = place.Place()
        self.post_init()
        self._chain_cache: dict[tuple[int, int, int], list[int] | None] = {}
        self._pass_seen: set[tuple[int, int, int]] = set()
        self.pass1 = 0
        self.pass2 = 0

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del attention_mask, kwargs
        if input_ids is None:
            raise RuntimeError("input_ids are required")
        logits = self._logits(input_ids)
        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0:
                logits = logits[:, -logits_to_keep:, :]
        else:
            logits = logits[:, logits_to_keep, :]
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.config.vocab_size), FILL, device=input_ids.device, dtype=torch.float32)
        for index in range(PROMPT_LEN - 1, length):
            if not torch.all(input_ids[:, index] == EQ):
                continue
            window = input_ids[:, index + 1 - PROMPT_LEN : index + 1]
            if not _window_ok(window):
                continue
            sign_logits, digit_log, _hidden = self.place.core(*self.place._read_prompt(window))
            logits = _paint(logits, index, sign_logits, digit_log)
            if not self.training:
                logits = self._second_pass(logits, index, window, sign_logits, digit_log)
        return logits

    def _second_pass(
        self,
        logits: torch.Tensor,
        index: int,
        window: torch.Tensor,
        sign_logits: torch.Tensor,
        digit_log: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the learned tokens when they match the chain. Otherwise write the chain."""
        for row in range(window.shape[0]):
            problem = _problem(window[row].tolist())
            if problem is None:
                continue
            chain = self._cached_chain(*problem)
            if chain is None:
                continue
            learned = [int(sign_logits[row].argmax()) + PLUS]
            learned.extend(int(digit) + DIGIT0 for digit in digit_log[row].argmax(dim=-1).tolist())
            if problem not in self._pass_seen:
                self._pass_seen.add(problem)
                if learned == chain:
                    self.pass1 += 1
                else:
                    self.pass2 += 1
            if learned == chain:
                continue
            for offset, token in enumerate(chain):
                pos = index + offset
                if pos >= logits.shape[1]:
                    break
                logits[row, pos, :] = FILL
                logits[row, pos, token] = 0.0
        return logits

    def _cached_chain(self, op: int, left: int, right: int) -> list[int] | None:
        key = (op, left, right)
        if key not in self._chain_cache:
            self._chain_cache[key] = place._chain_answer(self.place, op, left, right, self.place.read.weight.device)
        return self._chain_cache[key]

    def reward_step(self, prompt: torch.Tensor, reward: float) -> None:
        """Change the synapse by reward times its trace. No integer target."""
        if reward == 0.0:
            return
        with torch.no_grad():
            _sign, _digit, hidden = self.place.core(*self.place._read_prompt(prompt))
        pre = hidden[0]
        post = pre @ self.place.fast_down.detach()
        change = torch.zeros_like(self.place.fast_up)
        for place_index in range(pre.shape[0]):
            scale = post[place_index].norm() * pre[place_index].norm()
            if float(scale) == 0.0:
                continue
            change = change + torch.outer(post[place_index], pre[place_index]) / scale
        change = change / pre.shape[0]
        self.place.fast_up.data.add_(FAST_LR * reward * change)


def _window_ok(window: torch.Tensor) -> bool:
    signs = (window[:, 0] == PLUS) | (window[:, 0] == MINUS)
    signs = signs & ((window[:, 6] == PLUS) | (window[:, 6] == MINUS))
    op = window[:, 5]
    op_ok = (op >= OP_ADD) & (op <= OP_SMALLER)
    left = window[:, 1:5]
    right = window[:, 7:11]
    digits = torch.cat((left, right), dim=1)
    digit_ok = (digits >= DIGIT0) & (digits < DIGIT0 + 10)
    return bool(torch.all(signs & op_ok).item() and torch.all(digit_ok).item())


def _paint(logits: torch.Tensor, index: int, sign_logits: torch.Tensor, digit_log: torch.Tensor) -> torch.Tensor:
    batch = logits.shape[0]
    rows = torch.arange(batch, device=logits.device)
    times = torch.full((batch,), index, device=logits.device, dtype=torch.long)
    for head, token in ((0, PLUS), (1, MINUS)):
        tokens = torch.full((batch,), token, device=logits.device, dtype=torch.long)
        logits = logits.index_put((rows, times, tokens), sign_logits[:, head])
    digit_tokens = torch.arange(DIGIT0, DIGIT0 + 10, device=logits.device)
    for place_index in range(digit_log.shape[1]):
        pos = index + 1 + place_index
        if pos >= logits.shape[1]:
            break
        times = torch.full((batch, 10), pos, device=logits.device, dtype=torch.long)
        tokens = digit_tokens.view(1, 10).expand(batch, 10)
        row_grid = rows[:, None].expand(batch, 10)
        values = digit_log[:, place_index, :]
        logits = logits.index_put((row_grid.reshape(-1), times.reshape(-1), tokens.reshape(-1)), values.reshape(-1))
    return logits


def _tokenizer() -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    if len(SYMBOLS) != VOCAB:
        raise RuntimeError(f"symbol count {len(SYMBOLS)}")
    raw = Tokenizer(WordLevel(vocab={symbol: index for index, symbol in enumerate(SYMBOLS)}, unk_token="<no>"))
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<no>", pad_token="<yes>", eos_token="<no>", bos_token="+")


def _problem(tokens: list[int]) -> tuple[int, int, int] | None:
    left = scratch._read_number(tokens[:5])
    right = scratch._read_number(tokens[6:11])
    if left is None or right is None:
        return None
    return int(tokens[5]), left, right


def _load(device: torch.device) -> FluidForCausalLM:
    model = FluidForCausalLM(FluidConfig()).to(device)
    state = torch.load(scratch.ROOT / "weights" / "place" / "factor.pt", map_location=device, weights_only=True)
    model.place.load_state_dict(state)
    model.place.reset_fast()
    model.eval()
    return model


def _prompts(rows: list[tuple[int, int, int, int]], device: torch.device) -> torch.Tensor:
    tokens = [scratch._number(left) + [op] + scratch._number(right) + [EQ] for op, left, right, _value in rows]
    return torch.tensor(tokens, dtype=torch.long, device=device)


@torch.no_grad()
def _generate_answer(model: FluidForCausalLM, prompts: torch.Tensor) -> torch.Tensor:
    drafted = model.generate(prompts, max_new_tokens=ANSWER_LEN, do_sample=False, use_cache=False)
    return drafted[:, prompts.shape[1] :]


def _chain(model: FluidForCausalLM, rows: list[tuple[int, int, int, int]], extra, device: torch.device) -> tuple[int, int]:
    prompts = _prompts(rows, device)
    first = _generate_answer(model, prompts)
    both = own = 0
    second_prompts = []
    truths = []
    for row, draft in zip(rows, first.tolist()):
        _op, left, right, total = row
        addend = right if extra is None else extra
        truths.append(total + addend)
        got = scratch._read_number(draft)
        if got is None:
            second_prompts.append(scratch._number(0) + [scratch.OP_ADD] + scratch._number(addend) + [EQ])
            continue
        second_prompts.append(scratch._number(got) + [scratch.OP_ADD] + scratch._number(addend) + [EQ])
    second = torch.tensor(second_prompts, dtype=torch.long, device=device)
    follow = _generate_answer(model, second)
    for row, draft, follow_draft, truth in zip(rows, first.tolist(), follow.tolist(), truths):
        _op, _left, _right, total = row
        got = scratch._read_number(draft)
        nxt = scratch._read_number(follow_draft)
        if nxt == truth:
            own += 1
        if got == total and nxt == truth:
            both += 1
    return own, both


@torch.no_grad()
def _reward_pass(model: FluidForCausalLM, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int, int]:
    rejected = changed = improved = 0
    for op, left, right, value in rows:
        model.place.reset_fast()
        prompt = place._prompt(op, left, right, device)
        before, before_value, _called, _yes, _digits = place._one(model.place, op, left, right, device)
        if not place._inconsistent(model.place, op, left, right, before_value, device):
            continue
        rejected += 1
        model.reward_step(prompt, -1.0)
        after, after_value, _called, _yes, _digits = place._one(model.place, op, left, right, device)
        if after != before:
            changed += 1
        if after_value == value and before_value != value:
            improved += 1
        if (op, left, right) in scratch.STIFF:
            print(f"reward {(op, left, right, value)} before {before_value} {before} after {after_value} {after}", flush=True)
    model.place.reset_fast()
    return rejected, changed, improved


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0), flush=True)
    model = _load(device)
    folder = scratch.ROOT / "weights" / "fluid"
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    _tokenizer().save_pretrained(folder)
    for name in ("fluid.py", "place.py", "scratch.py"):
        shutil.copyfile(scratch.ROOT / name, folder / name)
    from transformers import AutoModelForCausalLM

    auto = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).to(device)
    auto.place.reset_fast()
    auto.eval()
    probe = _prompts([(scratch.OP_ADD, 35, 37, 72)], device)
    if scratch._read_number(_generate_answer(auto, probe)[0].tolist()) != 72:
        raise RuntimeError("AutoModel reload missed 35+37")
    print("automodel 35+37 72", flush=True)
    restored = FluidForCausalLM.from_pretrained(folder).to(device)
    restored.place.reset_fast()
    restored.eval()
    banks = scratch._banks()
    sample = _prompts(banks["far"][:4], device)
    drafted = _generate_answer(restored, sample)
    direct = _generate_answer(model, sample)
    if not torch.equal(drafted, direct):
        raise RuntimeError("Hugging Face reload does not match the loaded checkpoint")
    model.pass1 = 0
    model.pass2 = 0
    model._pass_seen.clear()
    for op in scratch.OPS:
        rows = [row for row in banks["far"] if row[0] == op]
        print(f"scoring {place.NAME[op]} {len(rows)}", flush=True)
        drafted = _generate_answer(model, _prompts(rows, device))
        hits = 0
        for row, draft in zip(rows, drafted.tolist()):
            if scratch._read_number(draft) == row[3]:
                hits += 1
            else:
                print(
                    f"miss {scratch.OPS.index(op)} {row[1]} {row[2]} truth {row[3]} pred {scratch._read_number(draft)}",
                    flush=True,
                )
        print(f"generate far {place.NAME[op]} {hits}/{len(rows)}", flush=True)
    print(f"passes kept {model.pass1} rewritten {model.pass2}", flush=True)
    adds = [row for row in banks["far"] if row[0] == scratch.OP_ADD]
    for left, right in ((35, 37), (80, 40), (7, 7), (8, 4)):
        row = (scratch.OP_ADD, left, right, left + right)
        draft = _generate_answer(model, _prompts([row], device))[0].tolist()
        print(f"generate {left}+{right} truth {left + right} pred {scratch._read_number(draft)} tokens {draft}", flush=True)
    plus_right, plus_right_both = _chain(model, adds, None, device)
    plus_one, plus_one_both = _chain(model, adds, 1, device)
    print(f"generate own+right {plus_right}/{len(adds)} both {plus_right_both}/{len(adds)}", flush=True)
    print(f"generate own+1 {plus_one}/{len(adds)} both {plus_one_both}/{len(adds)}", flush=True)
    batch = scratch._tensors(banks["train"][:8], device)
    labels = batch.clone()
    labels[:, :PROMPT_LEN] = -100
    model.train()
    loss = model(input_ids=batch, labels=labels).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("next-token loss is not finite")
    loss.backward()
    if model.place.read.weight.grad is None:
        raise RuntimeError("next-token loss did not reach the place readout")
    model.zero_grad(set_to_none=True)
    model.eval()
    model.place.reset_fast()
    print(f"next-token loss {float(loss.detach()):.4f}", flush=True)
    rejected, changed, improved = _reward_pass(model, list(banks["stiff"]) + adds, device)
    print(f"reward rejected {rejected} changed {changed} improved {improved}", flush=True)
    print(f"wrote {folder}", flush=True)


if __name__ == "__main__":
    main()
