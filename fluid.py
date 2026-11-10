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

``think`` writes a product as one addition at a time. Each line is a
``generate`` call. The next line's left side is the tokens that call just
wrote. A place shift is applied to those tokens, and the shifted tokens are
a later right operand. The command scores far products. It does not train
and it does not write a checkpoint.

``power`` multiplies that written product by the base again. A result that
would not fit in four places is not attempted. The last written power is
then divided by the base, and that quotient has to be the previous line.
The command does not train and it does not write a checkpoint.

``steps`` is the one runner for a new formula. A formula is a list of
operations on names. ``prev`` is the token list the previous line wrote.
The same runner multiplies, adds, and inverts. It does not train and it
does not write a checkpoint.

    python -u fluid.py
    python -u fluid.py think
    python -u fluid.py power
    python -u fluid.py steps
"""

from __future__ import annotations

import shutil
import sys

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


def _line_prompts(
    lefts: list[list[int]],
    rights: list[list[int]],
    op: int,
    device: torch.device,
) -> torch.Tensor:
    """One equation prompt. The left side is an answer already written."""
    if len(lefts) != len(rights) or not lefts:
        raise RuntimeError("thought lines are empty")
    rows = []
    for left, right in zip(lefts, rights):
        if len(left) != ANSWER_LEN or len(right) != ANSWER_LEN:
            raise RuntimeError("thought side is not an answer")
        rows.append(left + [op] + right + [EQ])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _write_lines(
    model: FluidForCausalLM,
    lefts: list[list[int]],
    rights: list[list[int]],
    op: int,
    device: torch.device,
) -> list[list[int]]:
    """Generate one line. The prompt's left tokens are the previous answer."""
    prompts = _line_prompts(lefts, rights, op, device)
    if prompts[:, :ANSWER_LEN].tolist() != lefts:
        raise RuntimeError("prompt does not start with the written answer")
    if prompts[:, 6 : 6 + ANSWER_LEN].tolist() != rights:
        raise RuntimeError("prompt right side drifted")
    return _generate_answer(model, prompts).tolist()


def _thought_products(
    model: FluidForCausalLM,
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> list[list[int]]:
    """Multiply by writing each addition and reading those tokens back."""
    count = len(pairs)
    left_digits = [place._operand_digits(left) for left, _right in pairs]
    right_digits = [place._operand_digits(right) for _left, right in pairs]
    partials: list[list[list[int]]] = [[] for _ in range(count)]
    probe = (37, 37)
    for i in range(place.PLACES):
        for j in range(place.PLACES):
            active = [
                row
                for row in range(count)
                if left_digits[row][i] != 0 and right_digits[row][j] != 0
            ]
            if not active:
                continue
            totals = [scratch._number(0) for _ in active]
            addends = [scratch._number(left_digits[row][i]) for row in active]
            remaining = [right_digits[row][j] for row in active]
            step = 0
            while any(item > 0 for item in remaining):
                todo = [index for index, item in enumerate(remaining) if item > 0]
                print(f"thought pair {i}+{j} step {step} rows {len(todo)}", flush=True)
                before = [totals[index] for index in todo]
                rights = [addends[index] for index in todo]
                wrote = _write_lines(model, before, rights, scratch.OP_ADD, device)
                for index, answer, left_tokens, right_tokens in zip(todo, wrote, before, rights):
                    if pairs[active[index]] == probe:
                        print(
                            f"  {scratch._read_number(left_tokens)}+{scratch._read_number(right_tokens)}={scratch._read_number(answer)}",
                            flush=True,
                        )
                    totals[index] = answer
                    remaining[index] -= 1
                step += 1
            for index, row in enumerate(active):
                shifted = place._answer(place._shift_places(place._digits_of(totals[index]), i + j))
                partials[row].append(shifted)
                if pairs[row] == probe:
                    print(
                        f"  place {i}+{j} -> {scratch._read_number(shifted)} tokens {shifted}",
                        flush=True,
                    )
    running = [scratch._number(0) for _ in range(count)]
    width = max((len(row) for row in partials), default=0)
    for index in range(width):
        todo = [row for row in range(count) if index < len(partials[row])]
        print(f"thought sum partial {index} rows {len(todo)}", flush=True)
        before = [running[row] for row in todo]
        rights = [partials[row][index] for row in todo]
        wrote = _write_lines(model, before, rights, scratch.OP_ADD, device)
        for row, answer, left_tokens, right_tokens in zip(todo, wrote, before, rights):
            if pairs[row] == probe:
                print(
                    f"  use {left_tokens}+{right_tokens}={answer}  {scratch._read_number(left_tokens)}+{scratch._read_number(right_tokens)}={scratch._read_number(answer)}",
                    flush=True,
                )
            running[row] = answer
    return running


def _report_thought() -> None:
    """Score far products whose steps are generate calls on the previous line."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0), flush=True)
    model = _load(device)
    rows = [row for row in scratch._banks()["far"] if row[0] == scratch.OP_MUL]
    pairs = [(left, right) for _op, left, right, _value in rows]
    written = _thought_products(model, pairs, device)
    hits = 0
    for row, tokens in zip(rows, written):
        got = scratch._read_number(tokens)
        if got == row[3]:
            hits += 1
        else:
            print(f"miss * {row[1]} {row[2]} truth {row[3]} pred {got}", flush=True)
        if (row[1], row[2]) == (37, 37):
            print(f"thought 37*37 = {got} tokens {tokens}", flush=True)
    print(f"thought far * {hits}/{len(rows)}", flush=True)
    plus = _write_lines(model, written, [scratch._number(1) for _ in rows], scratch.OP_ADD, device)
    plus_hits = 0
    for row, tokens, base in zip(rows, plus, written):
        if scratch._read_number(tokens) == row[3] + 1:
            plus_hits += 1
        else:
            print(
                f"miss *+1 {row[1]} {row[2]} truth {row[3] + 1} pred {scratch._read_number(tokens)}",
                flush=True,
            )
        if (row[1], row[2]) == (37, 37):
            print(
                f"thought {scratch._read_number(base)}+1 = {scratch._read_number(tokens)}",
                flush=True,
            )
    print(f"thought product+1 {plus_hits}/{len(rows)}", flush=True)
    divided = _write_lines(
        model,
        written,
        [scratch._number(right) for _op, _left, right, _value in rows],
        scratch.OP_DIV,
        device,
    )
    div_hits = 0
    for row, tokens in zip(rows, divided):
        if scratch._read_number(tokens) == row[1]:
            div_hits += 1
        else:
            print(f"miss */right {row[1]} {row[2]} truth {row[1]} pred {scratch._read_number(tokens)}", flush=True)
        if (row[1], row[2]) == (37, 37):
            print(f"thought product/37 = {scratch._read_number(tokens)}", flush=True)
    print(f"thought product/right {div_hits}/{len(rows)}", flush=True)


def _multiply_written(
    model: FluidForCausalLM,
    lefts: list[list[int]],
    rights: list[list[int]],
    device: torch.device,
    probe: set[tuple[int, int]] | None = None,
) -> list[list[int]]:
    """Multiply written answers. The next addition reads the tokens just written."""
    count = len(lefts)
    left_digits = [place._digits_of(tokens) for tokens in lefts]
    right_digits = [place._digits_of(tokens) for tokens in rights]
    partials: list[list[list[int]]] = [[] for _ in range(count)]
    watched = probe or set()
    for i in range(place.PLACES):
        for j in range(place.PLACES):
            active = [
                row
                for row in range(count)
                if left_digits[row][i] != 0 and right_digits[row][j] != 0
            ]
            if not active:
                continue
            print(f"power mul places {i}+{j} rows {len(active)}", flush=True)
            totals = [scratch._number(0) for _ in active]
            addends = [scratch._number(left_digits[row][i]) for row in active]
            remaining = [right_digits[row][j] for row in active]
            while any(item > 0 for item in remaining):
                todo = [index for index, item in enumerate(remaining) if item > 0]
                before = [totals[index] for index in todo]
                step_rights = [addends[index] for index in todo]
                wrote = _write_lines(model, before, step_rights, scratch.OP_ADD, device)
                for index, answer, left_tokens, right_tokens in zip(todo, wrote, before, step_rights):
                    pair = (scratch._read_number(lefts[active[index]]), scratch._read_number(rights[active[index]]))
                    if pair in watched:
                        print(
                            f"  {scratch._read_number(left_tokens)}+{scratch._read_number(right_tokens)}={scratch._read_number(answer)}",
                            flush=True,
                        )
                    totals[index] = answer
                    remaining[index] -= 1
            for index, row in enumerate(active):
                shifted = place._answer(place._shift_places(place._digits_of(totals[index]), i + j))
                partials[row].append(shifted)
                pair = (scratch._read_number(lefts[row]), scratch._read_number(rights[row]))
                if pair in watched:
                    print(
                        f"  place {i}+{j} -> {scratch._read_number(shifted)} tokens {shifted}",
                        flush=True,
                    )
    running = [scratch._number(0) for _ in range(count)]
    width = max((len(row) for row in partials), default=0)
    for index in range(width):
        todo = [row for row in range(count) if index < len(partials[row])]
        before = [running[row] for row in todo]
        step_rights = [partials[row][index] for row in todo]
        wrote = _write_lines(model, before, step_rights, scratch.OP_ADD, device)
        for row, answer, left_tokens, right_tokens in zip(todo, wrote, before, step_rights):
            pair = (scratch._read_number(lefts[row]), scratch._read_number(rights[row]))
            if pair in watched:
                print(
                    f"  use {left_tokens}+{right_tokens}={answer}  {scratch._read_number(left_tokens)}+{scratch._read_number(right_tokens)}={scratch._read_number(answer)}",
                    flush=True,
                )
            running[row] = answer
    return running


def _report_power() -> None:
    """Score powers built by multiplying the written line by the base again."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0), flush=True)
    model = _load(device)
    bases = list(range(2, scratch.FAR_HI + 1))
    limit = 10 ** scratch.WIDTH - 1
    base_tokens = [scratch._number(base) for base in bases]
    running = [list(tokens) for tokens in base_tokens]
    previous = [list(tokens) for tokens in base_tokens]
    square_hits = square_rows = 0
    higher_hits = higher_rows = 0
    shown = {2, 12, 37}
    probe = {(12, 12), (144, 12)}
    exponent = 1
    while True:
        todo = []
        for index, base in enumerate(bases):
            current = scratch._read_number(running[index])
            if current is None or current < 0 or current * base > limit:
                continue
            todo.append(index)
        if not todo:
            break
        exponent += 1
        print(f"power exp {exponent} rows {len(todo)}", flush=True)
        wrote = _multiply_written(
            model,
            [running[index] for index in todo],
            [base_tokens[index] for index in todo],
            device,
            probe,
        )
        for index, tokens in zip(todo, wrote):
            previous[index] = running[index]
            running[index] = tokens
            got = scratch._read_number(tokens)
            expect = bases[index] ** exponent
            if exponent == 2:
                square_rows += 1
                square_hits += int(got == expect)
            else:
                higher_rows += 1
                higher_hits += int(got == expect)
            if got != expect:
                print(f"miss ^{exponent} {bases[index]} truth {expect} pred {got}", flush=True)
            if bases[index] in shown:
                print(f"power {bases[index]}^{exponent} = {got} tokens {tokens}", flush=True)
    print(f"power squares {square_hits}/{square_rows}", flush=True)
    print(f"power higher {higher_hits}/{higher_rows}", flush=True)
    print("power divide-back", flush=True)
    quotients = _write_lines(model, running, base_tokens, scratch.OP_DIV, device)
    back_hits = 0
    for index, tokens in enumerate(quotients):
        got = scratch._read_number(tokens)
        expect = scratch._read_number(previous[index])
        if got == expect:
            back_hits += 1
        else:
            print(f"miss /base {bases[index]} truth {expect} pred {got}", flush=True)
        if bases[index] in shown:
            print(
                f"power {scratch._read_number(running[index])}/{bases[index]} = {got}",
                flush=True,
            )
    print(f"power divide-back {back_hits}/{len(bases)}", flush=True)


def _multiply_chain(
    model: FluidForCausalLM,
    lefts: list[list[int]],
    rights: list[list[int]],
    device: torch.device,
) -> list[list[int]]:
    """Multiply written answers with the place-value chain, one batch at a time."""
    count = len(lefts)
    left_digits = [place._digits_of(tokens) for tokens in lefts]
    right_digits = [place._digits_of(tokens) for tokens in rights]
    partials: list[list[list[int]]] = [[] for _ in range(count)]
    for i in range(place.PLACES):
        for j in range(place.PLACES):
            active = [
                row
                for row in range(count)
                if left_digits[row][i] != 0 and right_digits[row][j] != 0
            ]
            if not active:
                continue
            print(f"steps mul places {i}+{j} rows {len(active)}", flush=True)
            totals = [scratch._number(0) for _ in active]
            addends = [scratch._number(left_digits[row][i]) for row in active]
            remaining = [right_digits[row][j] for row in active]
            while any(item > 0 for item in remaining):
                todo = [index for index, item in enumerate(remaining) if item > 0]
                wrote = place._rule_add_many(
                    model.place,
                    [totals[index] for index in todo],
                    [addends[index] for index in todo],
                    device,
                )
                for index, answer in zip(todo, wrote):
                    totals[index] = answer
                    remaining[index] -= 1
            for index, row in enumerate(active):
                shifted = place._answer(place._shift_places(place._digits_of(totals[index]), i + j))
                partials[row].append(shifted)
    running = [scratch._number(0) for _ in range(count)]
    width = max((len(row) for row in partials), default=0)
    for index in range(width):
        todo = [row for row in range(count) if index < len(partials[row])]
        wrote = place._rule_add_many(
            model.place,
            [running[row] for row in todo],
            [partials[row][index] for row in todo],
            device,
        )
        for row, answer in zip(todo, wrote):
            running[row] = answer
    return running


def _apply_many(
    model: FluidForCausalLM,
    lefts: list[list[int]],
    rights: list[list[int]],
    op: int,
    device: torch.device,
) -> list[list[int]]:
    """One operation on written token lines."""
    if op == scratch.OP_ADD:
        return place._rule_add_many(model.place, lefts, rights, device)
    if op == scratch.OP_SUB:
        return place._rule_sub_many(model.place, lefts, rights, device)
    if op == scratch.OP_MUL:
        return _multiply_chain(model, lefts, rights, device)
    if op == scratch.OP_DIV:
        quotients, remainders = place._rule_div_many(model.place, lefts, rights, device)
        for remainder in remainders:
            if not place._is_zero(remainder):
                raise RuntimeError("a step division left a remainder")
        return quotients
    raise RuntimeError(f"step operation {op} is not in the runner")


def _run_steps(
    model: FluidForCausalLM,
    steps: tuple[tuple[str, str, int], ...],
    bound: dict[str, list[list[int]]],
    device: torch.device,
    prev: list[list[int]] | None = None,
    probe: int | None = None,
) -> list[list[int]]:
    """Run a formula. ``prev`` on the next line is the token list just written."""
    if not steps:
        raise RuntimeError("a formula has no steps")
    written = prev
    for left_name, right_name, op in steps:
        if left_name == "prev":
            if written is None:
                raise RuntimeError("a step asked for a line that was not written")
            lefts = written
        else:
            lefts = bound[left_name]
        if right_name == "prev":
            if written is None:
                raise RuntimeError("a step asked for a line that was not written")
            rights = written
        else:
            rights = bound[right_name]
        if left_name == "prev" and lefts is not written:
            raise RuntimeError("step left side is not the written line")
        if right_name == "prev" and rights is not written:
            raise RuntimeError("step right side is not the written line")
        wrote = _apply_many(model, lefts, rights, op, device)
        if probe is not None:
            print(
                f"  {scratch._read_number(lefts[probe])}{place.NAME[op]}{scratch._read_number(rights[probe])}={scratch._read_number(wrote[probe])} tokens {wrote[probe]}",
                flush=True,
            )
        written = wrote
    if written is None:
        raise RuntimeError("a formula wrote nothing")
    return written


def _score_lines(label: str, pairs: list[tuple[int, int]], written: list[list[int]], expect) -> None:
    hits = 0
    for (left, right), tokens in zip(pairs, written):
        got = scratch._read_number(tokens)
        truth = expect(left, right)
        if got == truth:
            hits += 1
        else:
            print(f"miss {label} {left} {right} truth {truth} pred {got}", flush=True)
    print(f"steps {label} {hits}/{len(pairs)}", flush=True)


def _report_steps() -> None:
    """Compose written lines with one runner, then invert those lines."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0), flush=True)
    model = _load(device)
    pairs = [(left, right) for left in range(scratch.FAR_LO, scratch.FAR_HI + 1) for right in range(scratch.FAR_LO, scratch.FAR_HI + 1)]
    bound = {
        "a": [scratch._number(left) for left, _right in pairs],
        "b": [scratch._number(right) for _left, right in pairs],
    }
    probe = pairs.index((37, 35))
    formulas = (
        (
            "sum*left",
            (("a", "b", scratch.OP_ADD), ("prev", "a", scratch.OP_MUL)),
            (("prev", "a", scratch.OP_DIV), ("prev", "b", scratch.OP_SUB)),
            lambda left, right: (left + right) * left,
            lambda left, _right: left,
        ),
        (
            "prod+left",
            (("a", "b", scratch.OP_MUL), ("prev", "a", scratch.OP_ADD)),
            (("prev", "a", scratch.OP_SUB), ("prev", "a", scratch.OP_DIV)),
            lambda left, right: (left * right) + left,
            lambda _left, right: right,
        ),
    )
    for name, forward, inverse, forward_expect, inverse_expect in formulas:
        print(f"steps {name}", flush=True)
        written = _run_steps(model, forward, bound, device, probe=probe)
        _score_lines(name, pairs, written, forward_expect)
        print(f"steps {name} back", flush=True)
        restored = _run_steps(model, inverse, bound, device, prev=written, probe=probe)
        _score_lines(f"{name} back", pairs, restored, inverse_expect)
    print("steps 12^3", flush=True)
    twelve = {"a": [scratch._number(12)]}
    cubed = _run_steps(
        model,
        (("a", "a", scratch.OP_MUL), ("prev", "a", scratch.OP_MUL)),
        twelve,
        device,
        probe=0,
    )
    print(f"steps 12^3 = {scratch._read_number(cubed[0])}", flush=True)
    if scratch._read_number(cubed[0]) != 12 ** 3:
        raise RuntimeError("the step runner missed 12^3")
    restored = _run_steps(
        model,
        (("prev", "a", scratch.OP_DIV), ("prev", "a", scratch.OP_DIV)),
        twelve,
        device,
        prev=cubed,
        probe=0,
    )
    print(f"steps 12^3 back = {scratch._read_number(restored[0])}", flush=True)
    if scratch._read_number(restored[0]) != 12:
        raise RuntimeError("the step runner did not invert 12^3")


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
    if len(sys.argv) > 1 and sys.argv[1] == "think":
        _report_thought()
    elif len(sys.argv) > 1 and sys.argv[1] == "power":
        _report_power()
    elif len(sys.argv) > 1 and sys.argv[1] == "steps":
        _report_steps()
    else:
        main()
