"""Owned place forward.

One mix runs on every pair of digits. A sweep then walks from the least
place to the greatest. Addition, subtraction, comparison, and division
name one total at that place. The carry is an offset of that total: 12 is
digit 2 and carry 1, and the next place adds the 1. The sweep does not see
the carry, so a learned vector cannot drop it.

Multiplication does not use that blend, and it does not use that mix.
Each nonzero digit pair names its own product. A zero digit contributes
product 0, so the head cannot rename it. Pairs on one place are added as
distributions, and the carry is the offset of that sum. There is no place
index in the product head. A pair learned on a small product is the same
pair at a larger magnitude.

A study pass rehearses one digit pair at a time. It rebuilds the pair from
smaller pairs it already repeats and adds those with its own addition. A sum
is kept when its digits are the place-value offset of the two parts, the
same offset the carry uses. One order is enough when the other breaks that
offset. One such rebuild replaces the direct product. A pair with no matching
rebuild keeps the learned product. One times a digit is that digit, the same
kind of gate as a zero digit. The note is not a slow weight, and the offset
check does not supply the sum.

A second pass checks a draft by running the inverse operation through this
same forward. A comparison that is neither operand is checked the same way.
The outside answer is only a score.

The think path does not read the product head. Each addition is written as
answer tokens, and the next addition reads those tokens. A digit product is
that digit added, count times, then shifted by place. The partials are added
the same way. Subtraction is the same offset in the other direction. A final
borrow is a negative sign, and the magnitude is the place-value complement.
Comparison reads that sign. Division subtracts until the remainder would
change sign, and the count of those subtractions is the quotient. The written
product is then divided by each factor.
``python -u place.py think`` scores that chain and does not write a checkpoint.

``python -u place.py learn`` teaches the sweep that place total. The lesson is
the offset of the two digits, not the far answer. Multiplication weights are
left alone. The far score is taken once, after training.

The training banks, seed, and keep-by-far-count rule are the ones in
``scratch.py``. This run writes ``weights/place/`` only.

    python -u place.py
"""

from __future__ import annotations

import json
import shutil
import sys

import torch
import torch.nn.functional as F
from torch import nn

import scratch

BASE = 10
PLACES = scratch.WIDTH
H = PLACES * BASE
RANK = BASE
VMIN = -20
VMAX = 99
NCONTRIB = VMAX - VMIN + 1
CARRY_MAX = 10
CARRY = CARRY_MAX + 1
NPROD = 82  # a digit product is 0..81
MUL_CARRY = 40  # 4 * 81 + 39 = 363 and 363 // 10 = 36, inside 0..39
YES = scratch.YES
NO = scratch.NO
PLUS_SIGN = scratch.PLUS_SIGN
MINUS_SIGN = scratch.MINUS_SIGN
DIGIT0 = scratch.DIGIT0
OP_ADD = scratch.OP_ADD
PROMPT_LEN = scratch.PROMPT_LEN
DEST = scratch.ROOT / "weights" / "place"
INV = {
    scratch.OP_ADD: scratch.OP_SUB,
    scratch.OP_SUB: scratch.OP_ADD,
    scratch.OP_MUL: scratch.OP_DIV,
    scratch.OP_DIV: scratch.OP_MUL,
}
NAME = {
    scratch.OP_ADD: "+",
    scratch.OP_SUB: "-",
    scratch.OP_MUL: "*",
    scratch.OP_DIV: "/",
    scratch.OP_LARGER: "larger",
    scratch.OP_SMALLER: "smaller",
}


def _convolve(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Distribution of a sum. Exact, so the next place can still see the carry."""
    wide = left[:, :, None] * right[:, None, :]
    left_index = torch.arange(left.shape[1], device=left.device)
    right_index = torch.arange(right.shape[1], device=left.device)
    index = (left_index[:, None] + right_index[None, :]).reshape(-1)
    out = left.new_zeros(left.shape[0], left.shape[1] + right.shape[1] - 1)
    return out.index_add(1, index, wide.reshape(left.shape[0], -1))


def _slow_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in model.named_parameters() if not name.startswith("fast_")]


class Place(nn.Module):
    """Shared place mix, shared carry sweep, silent short-term residual."""

    def __init__(self) -> None:
        super().__init__()
        self.digit = nn.Embedding(BASE, H)
        self.sign = nn.Embedding(2, H)
        self.op = nn.Embedding(len(scratch.OPS), H)
        self.mix1 = nn.Linear(5 * H, H)
        self.mix2 = nn.Linear(H, H)
        self.sweep1 = nn.Linear(4 * H, H)
        self.sweep2 = nn.Linear(H, H)
        self.read = nn.Linear(H, NCONTRIB)
        self.sign_carry = nn.Linear(CARRY, H)
        self.sign_read = nn.Linear(4 * H, 2)
        self.place_verdict = nn.Linear(2 * H, 2)
        self.eq_verdict = nn.Linear(3 * H, 2)
        self.fast_down = nn.Parameter(torch.empty(H, RANK))
        self.fast_up = nn.Parameter(torch.zeros(RANK, H))
        nn.init.normal_(self.fast_down, std=0.02)
        self.register_buffer("fast_down0", self.fast_down.detach().clone())
        # After the sweep init, so this fresh product mix does not reshuffle it.
        self.prod_digit = nn.Embedding(BASE, H)
        self.prod_sign = nn.Embedding(2, H)
        self.prod_op = nn.Embedding(len(scratch.OPS), H)
        self.prod_mix1 = nn.Linear(5 * H, H)
        self.prod_mix2 = nn.Linear(H, H)
        self.product = nn.Linear(H, NPROD)
        self.mul_sign_carry = nn.Linear(MUL_CARRY, H)

    def reset_fast(self) -> None:
        self.fast_down.data.copy_(self.fast_down0)
        self.fast_up.data.zero_()

    def _mlp(self, first: nn.Linear, second: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        return second(F.gelu(first(hidden)))

    def _fast(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + (hidden @ self.fast_down @ self.fast_up)

    def _read_prompt(self, prompt: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if prompt.shape[1] != PROMPT_LEN:
            raise RuntimeError(f"prompt length {prompt.shape[1]}")
        left_sign = prompt[:, 0] - PLUS_SIGN
        left_digits = prompt[:, 1:5] - DIGIT0
        op = prompt[:, 5] - OP_ADD
        right_sign = prompt[:, 6] - PLUS_SIGN
        right_digits = prompt[:, 7:11] - DIGIT0
        return left_sign, left_digits, op, right_sign, right_digits

    def _shift(self, contrib: torch.Tensor, carry: torch.Tensor, sign: int) -> torch.Tensor:
        low = VMIN - CARRY_MAX
        high = VMAX + CARRY_MAX
        out = contrib.new_zeros(contrib.shape[0], high - low + 1)
        values = torch.arange(VMIN, VMAX + 1, device=contrib.device)
        for step in range(CARRY):
            dest = values + sign * step - low
            out = out.index_add(1, dest, contrib * carry[:, step : step + 1])
        return out

    def _combine(self, contrib: torch.Tensor, carry: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the carry as an offset. Direction is +1, -1, or 0."""
        batch = contrib.shape[0]
        zero = F.one_hot(torch.zeros(batch, dtype=torch.long, device=contrib.device), CARRY).to(contrib.dtype)
        up = self._shift(contrib, carry, 1)
        down = self._shift(contrib, carry, -1)
        same = self._shift(contrib, zero, 1)
        mark = direction.view(batch, 1)
        total = torch.where(mark > 0, up, torch.where(mark < 0, down, same))
        low = VMIN - CARRY_MAX
        values = torch.arange(low, VMAX + CARRY_MAX + 1, device=contrib.device)
        digit = total.new_zeros(batch, BASE).index_add(1, values % 10, total)
        quotient = values // 10
        next_add = total.new_zeros(batch, CARRY).index_add(1, quotient.clamp(0, CARRY_MAX), total)
        next_sub = total.new_zeros(batch, CARRY).index_add(1, (-quotient).clamp(0, CARRY_MAX), total)
        next_carry = torch.where(mark > 0, next_add, torch.where(mark < 0, next_sub, zero))
        return digit.clamp_min(1e-8).log(), next_carry

    def _mul_digit(self, total: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.arange(total.shape[1], device=total.device)
        digit = total.new_zeros(total.shape[0], BASE).index_add(1, values % BASE, total)
        nxt = total.new_zeros(total.shape[0], MUL_CARRY).index_add(1, (values // BASE).clamp(max=MUL_CARRY - 1), total)
        return digit.clamp_min(1e-8).log(), nxt

    def _pair_mix(
        self,
        left_e: torch.Tensor,
        right_e: torch.Tensor,
        op_e: torch.Tensor,
        sign_l: torch.Tensor,
        sign_r: torch.Tensor,
        first: nn.Linear,
        second: nn.Linear,
    ) -> torch.Tensor:
        batch, places, hidden = left_e.shape
        left_pair = left_e[:, :, None, :].expand(batch, places, places, hidden)
        right_pair = right_e[:, None, :, :].expand(batch, places, places, hidden)
        op_pair = op_e[:, None, None, :].expand(batch, places, places, hidden)
        sign_l_pair = sign_l[:, None, None, :].expand(batch, places, places, hidden)
        sign_r_pair = sign_r[:, None, None, :].expand(batch, places, places, hidden)
        packed = torch.cat((left_pair, right_pair, op_pair, sign_l_pair, sign_r_pair), dim=-1)
        mixed = self._mlp(first, second, packed.reshape(batch * places * places, 5 * hidden))
        return mixed.reshape(batch, places, places, hidden)

    def _gate(self, learned: torch.Tensor, left_digits: torch.Tensor, right_digits: torch.Tensor) -> torch.Tensor:
        """A zero digit contributes product 0. A studied pair contributes its note."""
        zero_pair = (left_digits[:, :, None] == 0) | (right_digits[:, None, :] == 0)
        one = learned.new_zeros(learned.shape)
        one[..., 0] = 1
        gated = torch.where(zero_pair.unsqueeze(-1), one, learned)
        note = getattr(self, "study_note", None)
        if note is None:
            return gated
        choice = note[left_digits[:, :, None], right_digits[:, None, :]]
        use = (choice >= 0).unsqueeze(-1) & ~zero_pair.unsqueeze(-1)
        eye = learned.new_zeros(learned.shape)
        eye.scatter_(-1, choice.clamp(min=0).unsqueeze(-1), 1)
        return torch.where(use, eye, gated)

    def _from_pairs(self, dist: torch.Tensor, fired: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, places, _, hidden = fired.shape
        carry = F.one_hot(torch.zeros(batch, dtype=torch.long, device=dist.device), MUL_CARRY).to(dtype=dist.dtype)
        hiddens = []
        logits = []
        for place in range(places):
            parts = []
            acc = fired.new_zeros(batch, hidden)
            for i in range(places):
                j = place - i
                if 0 <= j < places:
                    parts.append(dist[:, i, j, :])
                    acc = acc + fired[:, i, j, :]
            total = parts[0]
            for part in parts[1:]:
                total = _convolve(total, part)
            total = _convolve(total, carry)
            digit_log, carry = self._mul_digit(total)
            hiddens.append(acc)
            logits.append(digit_log)
        return torch.stack(logits, dim=1), torch.stack(hiddens, dim=1), carry

    def _multiply(
        self,
        left_digits: torch.Tensor,
        right_digits: torch.Tensor,
        op: torch.Tensor,
        left_sign: torch.Tensor,
        right_sign: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Name a product per nonzero digit pair, then add the pairs that share a place."""
        mixed = self._pair_mix(
            self.prod_digit(left_digits),
            self.prod_digit(right_digits),
            self.prod_op(op),
            self.prod_sign(left_sign),
            self.prod_sign(right_sign),
            self.prod_mix1,
            self.prod_mix2,
        )
        batch, places, _, hidden = mixed.shape
        fired = self._fast(mixed.reshape(batch * places * places, hidden))
        learned = torch.softmax(self.product(fired), dim=-1).reshape(batch, places, places, NPROD)
        fired = fired.reshape(batch, places, places, hidden)
        return self._from_pairs(self._gate(learned, left_digits, right_digits), fired)

    def core(
        self,
        left_sign: torch.Tensor,
        left_digits: torch.Tensor,
        op: torch.Tensor,
        right_sign: torch.Tensor,
        right_digits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_e = self.digit(left_digits)
        right_e = self.digit(right_digits)
        op_e = self.op(op)
        sign_l = self.sign(left_sign)
        sign_r = self.sign(right_sign)
        batch, places, hidden = left_e.shape
        mixed = self._pair_mix(left_e, right_e, op_e, sign_l, sign_r, self.mix1, self.mix2)
        mul_logits, mul_hidden, mul_carry = self._multiply(left_digits, right_digits, op, left_sign, right_sign)
        slots = places * 2 - 1
        pair_index = torch.arange(places, device=mixed.device)
        pair_index = (pair_index[:, None] + pair_index[None, :]).reshape(-1)
        cross = torch.zeros(batch, slots, hidden, device=mixed.device, dtype=mixed.dtype)
        cross = cross.index_add(1, pair_index, mixed.reshape(batch, places * places, hidden))
        index = torch.arange(places, device=mixed.device)
        diagonal = mixed[:, index, index, :]
        equal = ((op == 0) | (op == 1) | (op == 4) | (op == 5)).to(diagonal.dtype)
        gathered = equal[:, None, None] * diagonal + (1.0 - equal[:, None, None]) * cross[:, :places]
        direction = torch.zeros(batch, dtype=torch.long, device=op.device)
        direction = torch.where((op == 0) | (op == 2), torch.ones_like(op), direction)
        direction = torch.where((op == 1) | (op == 3), -torch.ones_like(op), direction)
        carry = F.one_hot(torch.zeros(batch, dtype=torch.long, device=op.device), CARRY).to(diagonal.dtype)
        hiddens = []
        logits = []
        read_steps = []
        for place in range(places):
            step_in = torch.cat((gathered[:, place], op_e, sign_l, sign_r), dim=-1)
            hidden_p = self._fast(self._mlp(self.sweep1, self.sweep2, step_in))
            hiddens.append(hidden_p)
            read_logit = self.read(hidden_p)
            contrib = torch.softmax(read_logit, dim=-1)
            digit_log, carry = self._combine(contrib, carry, direction)
            logits.append(digit_log)
            read_steps.append(read_logit)
        self._read_logits = torch.stack(read_steps, dim=1)
        add_hidden = torch.stack(hiddens, dim=1)
        add_logits = torch.stack(logits, dim=1)
        mark = (op == 2).view(batch, 1, 1)
        hidden_stack = torch.where(mark, mul_hidden, add_hidden)
        digit_logits = torch.where(mark, mul_logits, add_logits)
        carry_h = torch.where((op == 2).view(batch, 1), self.mul_sign_carry(mul_carry), self.sign_carry(carry))
        sign_in = torch.cat((hidden_stack.sum(dim=1), carry_h, sign_l, sign_r), dim=-1)
        return self.sign_read(sign_in), digit_logits, hidden_stack

    def judge(self, hidden: torch.Tensor, digits: torch.Tensor, sign_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.digit(digits)
        place_logits = self.place_verdict(torch.cat((hidden, embedded), dim=-1))
        equation = torch.cat((hidden.mean(dim=1), self.sign(sign_index), embedded.mean(dim=1)), dim=-1)
        return place_logits, self.eq_verdict(equation)

    def predict_bundle(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sign_logits, digit_logits, hidden = self.core(*self._read_prompt(prompt))
        sign = sign_logits.argmax(dim=-1) + PLUS_SIGN
        digits = digit_logits.argmax(dim=-1) + DIGIT0
        tokens = torch.cat((sign.unsqueeze(1), digits), dim=1)
        return tokens, sign_logits, digit_logits, hidden


def _answer_index(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sign_index = (tokens[:, PROMPT_LEN] == MINUS_SIGN).long()
    digits = tokens[:, PROMPT_LEN + 1 :] - DIGIT0
    return sign_index, digits


def _prompt(op: int, left: int, right: int, device: torch.device) -> torch.Tensor:
    tokens = scratch._number(left) + [op] + scratch._number(right) + [scratch.EQ]
    if len(tokens) != PROMPT_LEN:
        raise RuntimeError("prompt length drifted")
    return torch.tensor([tokens], dtype=torch.long, device=device)


def _loss(model: Place, batch: torch.Tensor, forged: torch.Tensor) -> torch.Tensor:
    prompt = batch[:, :PROMPT_LEN]
    sign_logits, digit_logits, hidden = model.core(*model._read_prompt(prompt))
    true_sign, true_digits = _answer_index(batch)
    forged_sign, forged_digits = _answer_index(forged)
    loss = F.nll_loss(digit_logits.reshape(-1, BASE), true_digits.reshape(-1))
    loss = loss + F.cross_entropy(sign_logits, true_sign)
    place_true, eq_true = model.judge(hidden, true_digits, true_sign)
    place_forged, eq_forged = model.judge(hidden, forged_digits, forged_sign)
    yes = torch.full((len(batch),), YES, device=batch.device, dtype=torch.long)
    no = torch.full((len(batch),), NO, device=batch.device, dtype=torch.long)
    yes_places = torch.full((len(batch), PLACES), YES, device=batch.device, dtype=torch.long)
    loss = loss + 0.5 * F.cross_entropy(eq_true, yes) + 0.5 * F.cross_entropy(eq_forged, no)
    loss = loss + 0.5 * F.cross_entropy(place_true.reshape(-1, 2), yes_places.reshape(-1))
    place_target = torch.where(forged_digits == true_digits, YES, NO)
    loss = loss + 0.5 * F.cross_entropy(place_forged.reshape(-1, 2), place_target.reshape(-1))
    return loss


@torch.no_grad()
def _exact(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int]:
    if not rows:
        return 0, 0
    hits = 0
    for start in range(0, len(rows), scratch.BATCH):
        chunk = rows[start : start + scratch.BATCH]
        prompt = scratch._tensors(chunk, device)[:, :PROMPT_LEN]
        drafted, _sign, _digits, _hidden = model.predict_bundle(prompt)
        for row, draft in zip(chunk, drafted.tolist()):
            if scratch._read_number(draft) == row[3]:
                hits += 1
    return hits, len(rows)


def _by_op(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> str:
    groups = {op: [] for op in scratch.OPS}
    for row in rows:
        groups[row[0]].append(row)
    return " ".join(f"{NAME[op]} {_exact(model, groups[op], device)[0]}/{len(groups[op])}" for op in scratch.OPS)


@torch.no_grad()
def _verdict_hits(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int]:
    hits = 0
    for start in range(0, len(rows), scratch.BATCH):
        chunk = rows[start : start + scratch.BATCH]
        prompt = scratch._tensors(chunk, device)[:, :PROMPT_LEN]
        drafted, _sign, _digits, hidden = model.predict_bundle(prompt)
        sign_index = (drafted[:, 0] == MINUS_SIGN).long()
        _place, equation = model.judge(hidden, drafted[:, 1:] - DIGIT0, sign_index)
        called = equation.argmax(dim=-1).tolist()
        for row, draft, call in zip(chunk, drafted.tolist(), called):
            right = scratch._read_number(draft) == row[3]
            if (call == YES and right) or (call == NO and not right):
                hits += 1
    return hits, len(rows)


@torch.no_grad()
def _verdict_repair_hits(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int]:
    """Where a place judge says no, take the digit that judge prefers."""
    hits = 0
    for start in range(0, len(rows), scratch.BATCH):
        chunk = rows[start : start + scratch.BATCH]
        prompt = scratch._tensors(chunk, device)[:, :PROMPT_LEN]
        drafted, _sign, _digits, hidden = model.predict_bundle(prompt)
        digits = drafted[:, 1:] - DIGIT0
        place_logits, _equation = model.judge(hidden, digits, (drafted[:, 0] == MINUS_SIGN).long())
        weight = model.digit.weight
        batch, places, hidden_size = hidden.shape
        left = hidden[:, :, None, :].expand(batch, places, BASE, hidden_size)
        right = weight[None, None, :, :].expand(batch, places, BASE, hidden_size)
        scores = model.place_verdict(torch.cat((left, right), dim=-1))
        preferred = scores[:, :, :, YES].argmax(dim=-1)
        chosen = torch.where(place_logits.argmax(dim=-1) == NO, preferred, digits)
        for row, sign_token, digs in zip(chunk, drafted[:, 0].tolist(), chosen.tolist()):
            answer = [sign_token, *[digit + DIGIT0 for digit in digs]]
            if scratch._read_number(answer) == row[3]:
                hits += 1
    return hits, len(rows)


def _agree(got: int | None, target: int) -> int:
    if got is None:
        return 0
    return sum(left == right for left, right in zip(scratch._number(got), scratch._number(target)))


@torch.no_grad()
def _one(model: Place, op: int, left: int, right: int, device: torch.device) -> tuple[list[int], int | None, bool, float, torch.Tensor]:
    prompt = _prompt(op, left, right, device)
    tokens, _sign, digit_logits, hidden = model.predict_bundle(prompt)
    drafted = tokens[0].tolist()
    value = scratch._read_number(drafted)
    sign_index = torch.tensor([1 if drafted[0] == MINUS_SIGN else 0], device=device)
    _place, equation = model.judge(hidden, tokens[:, 1:] - DIGIT0, sign_index)
    yes = float(equation[0, YES].detach())
    return drafted, value, int(equation.argmax()) == YES, yes, digit_logits


def _candidates(model: Place, op: int, left: int, right: int, device: torch.device) -> list[list[int]]:
    drafted, _value, _called, _yes, digit_logits = _one(model, op, left, right, device)
    found = []
    flipped = drafted.copy()
    flipped[0] = MINUS_SIGN if flipped[0] == PLUS_SIGN else PLUS_SIGN
    found.append(flipped)
    prompt = _prompt(op, left, right, device)
    _tokens, _sign, _digit_logits, hidden = model.predict_bundle(prompt)
    for place in range(PLACES):
        order = torch.topk(digit_logits[0, place], k=2)
        second = drafted.copy()
        second[1 + place] = DIGIT0 + int(order.indices[1])
        found.append(second)
        weight = model.digit.weight
        step = hidden[0, place][None, :].expand(BASE, H)
        scores = model.place_verdict(torch.cat((step, weight), dim=-1))
        preferred = drafted.copy()
        preferred[1 + place] = DIGIT0 + int(scores[:, YES].argmax())
        found.append(preferred)
    return found


def _search(model: Place, op: int, left: int, right: int, device: torch.device) -> list[int]:
    drafted, value, _called, yes, _digits = _one(model, op, left, right, device)
    best = drafted
    inverse = INV.get(op)
    if inverse is not None and not (inverse == scratch.OP_DIV and right == 0) and value is not None:
        recovered, _rec_value, _rec_yes, _rec_logit, _rec_digits = _one(model, inverse, value, right, device)
        best_agree = _agree(scratch._read_number(recovered), left)
        for candidate in _candidates(model, op, left, right, device):
            candidate_value = scratch._read_number(candidate)
            if candidate_value is None:
                continue
            recovered, _a, _b, _c, _d = _one(model, inverse, candidate_value, right, device)
            agree = _agree(scratch._read_number(recovered), left)
            if agree > best_agree:
                best_agree = agree
                best = candidate
        return best
    best_yes = yes
    for candidate in _candidates(model, op, left, right, device):
        prompt_value = scratch._read_number(candidate)
        if prompt_value is None:
            continue
        _draft, _value, _called, candidate_yes, _digits = _one(model, op, left, right, device)
        # The candidate is an edited draft, not a new question. Score it with the judge
        # on the original hidden state.
        prompt = _prompt(op, left, right, device)
        _tokens, _sign, _digit_logits, hidden = model.predict_bundle(prompt)
        sign_index = torch.tensor([1 if candidate[0] == MINUS_SIGN else 0], device=device)
        digits = torch.tensor([[token - DIGIT0 for token in candidate[1:]]], device=device)
        _place, equation = model.judge(hidden, digits, sign_index)
        candidate_yes = float(equation[0, YES].detach())
        if candidate_yes > best_yes:
            best_yes = candidate_yes
            best = candidate
    return best


def _fast_step(model: Place, op: int, left: int, right: int, target: list[int], device: torch.device) -> None:
    sign_index = torch.tensor([1 if target[0] == MINUS_SIGN else 0], device=device)
    digits = torch.tensor([[token - DIGIT0 for token in target[1:]]], device=device)
    optimizer = torch.optim.SGD([model.fast_down, model.fast_up], lr=0.1)
    model.zero_grad(set_to_none=True)
    prompt = _prompt(op, left, right, device)
    sign_logits, digit_logits, _hidden = model.core(*model._read_prompt(prompt))
    loss = F.nll_loss(digit_logits.reshape(-1, BASE), digits.reshape(-1))
    loss = loss + F.cross_entropy(sign_logits, sign_index)
    loss.backward()
    optimizer.step()


def _inconsistent(model: Place, op: int, left: int, right: int, value: int | None, device: torch.device) -> bool:
    """A draft fails when the inverse does not return the left operand, or a comparison is neither operand."""
    inverse = INV.get(op)
    if inverse is None:
        return value not in (left, right)
    if value is None or (inverse == scratch.OP_DIV and right == 0):
        return True
    _tokens, recovered, _called, _yes, _digits = _one(model, inverse, value, right, device)
    return recovered != left


@torch.no_grad()
def _operand_choice(model: Place, op: int, left: int, right: int, device: torch.device) -> list[int]:
    """Between the two operands, keep the one this forward gives the higher log probability."""
    _tokens, sign_logits, digit_logits, _hidden = model.predict_bundle(_prompt(op, left, right, device))
    sign_log = torch.log_softmax(sign_logits, dim=-1)
    digit_log = torch.log_softmax(digit_logits, dim=-1)

    def score(value: int) -> float:
        number = scratch._number(value)
        total = sign_log[0, 0 if value >= 0 else 1]
        for place, token in enumerate(number[1:]):
            total = total + digit_log[0, place, token - DIGIT0]
        return float(total.detach())

    chosen = left if score(left) >= score(right) else right
    return scratch._number(chosen)


def _revise(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int, int]:
    rejected = changed = improved = 0
    for op, left, right, value in rows:
        model.reset_fast()
        before, before_value, _called, _yes, _digits = _one(model, op, left, right, device)
        if not _inconsistent(model, op, left, right, before_value, device):
            continue
        rejected += 1
        if INV.get(op) is None:
            alternative = _operand_choice(model, op, left, right, device)
        else:
            alternative = _search(model, op, left, right, device)
        if alternative == before:
            continue
        _fast_step(model, op, left, right, alternative, device)
        after, after_value, _called, _yes, _digits = _one(model, op, left, right, device)
        if after != before:
            changed += 1
        if after_value == value and before_value != value:
            improved += 1
    model.reset_fast()
    return rejected, changed, improved


def _offset_agrees(left: int, right: int, total: int) -> bool:
    """True when the written total is the place-value offset of the two parts.

    This does not return a sum. A rebuild the model wrote is dropped when
    its digits disagree with that offset.
    """
    if left < 0 or right < 0 or total < 0:
        return False
    carry = 0
    for _ in range(6):
        column = (left % 10) + (right % 10) + carry
        if total % 10 != column % 10:
            return False
        carry = column // 10
        left //= 10
        right //= 10
        total //= 10
    return carry == 0 and total == 0


def _model_sum(model: Place, left: int, right: int, device: torch.device, cache: dict[tuple[int, int, int], int | None]) -> int | None:
    """The model's addition, kept when a written order matches the place-value offset."""

    def ask(op: int, one: int, other: int) -> int | None:
        key = (op, one, other)
        if key not in cache:
            _tokens, value, _called, _yes, _digits = _one(model, op, one, other, device)
            cache[key] = value
        return cache[key]

    passed: list[int] = []
    for value in (ask(scratch.OP_ADD, left, right), ask(scratch.OP_ADD, right, left)):
        if value is None or not 0 <= value <= 81 or not _offset_agrees(left, right, value):
            continue
        if value not in passed:
            passed.append(value)
    if len(passed) == 1:
        return passed[0]
    return None


def _split_sums(factor: int, other: int, notes: dict[tuple[int, int], int], model: Place, device: torch.device, cache: dict[tuple[int, int, int], int | None]) -> list[int]:
    """Rebuild other * factor from parts already in the notebook."""
    found = []
    if factor < 2:
        return found
    for part in range(1, factor):
        rest = factor - part
        if part > rest:
            continue
        if (other, part) not in notes or (other, rest) not in notes:
            continue
        total = _model_sum(model, notes[(other, part)], notes[(other, rest)], device, cache)
        if total is not None:
            found.append(total)
    return found


@torch.no_grad()
def _study(model: Place, device: torch.device) -> None:
    """Rehearse digit pairs until a rebuild sticks, then keep that note on the pair."""
    if not _offset_agrees(24, 18, 42) or _offset_agrees(3, 6, 19) or _offset_agrees(28, 28, 57):
        raise RuntimeError("place-value witness drifted")
    model.eval()
    model.reset_fast()
    model.study_note = None
    cache: dict[tuple[int, int, int], int | None] = {}
    direct: dict[tuple[int, int], int | None] = {}
    for left in range(BASE):
        for right in range(BASE):
            _tokens, value, _called, _yes, _digits = _one(model, scratch.OP_MUL, left, right, device)
            direct[(left, right)] = value
            cache[(scratch.OP_MUL, left, right)] = value
    notes: dict[tuple[int, int], int] = {}
    for digit in range(BASE):
        notes[(0, digit)] = 0
        notes[(digit, 0)] = 0
        notes[(1, digit)] = digit
        notes[(digit, 1)] = digit
    for larger in range(2, BASE):
        for smaller in range(2, larger + 1):
            votes = _split_sums(larger, smaller, notes, model, device, cache)
            if smaller != larger:
                votes = votes + _split_sums(smaller, larger, notes, model, device, cache)
            if not votes:
                continue
            learned = direct[(smaller, larger)]
            if learned is None:
                continue
            counts: dict[int, int] = {}
            for vote in votes:
                counts[vote] = counts.get(vote, 0) + 1
            best = max(counts, key=counts.get)
            count = counts[best]
            second = max((seen for value, seen in counts.items() if value != best), default=0)
            if count > second:
                notes[(smaller, larger)] = best
                notes[(larger, smaller)] = best
                if best != learned:
                    print(f"study {smaller}*{larger} {learned} -> {best}", flush=True)
    table = torch.full((BASE, BASE), -1, device=device, dtype=torch.long)
    for (left, right), value in notes.items():
        if not 0 <= value < NPROD:
            raise RuntimeError(f"study note {left}*{right}={value} is outside 0..81")
        table[left, right] = value
    model.study_note = table


def _fit(model: Place, train: list[tuple[int, int, int, int]], banks: dict[str, list[tuple[int, int, int, int]]], device: torch.device) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(scratch.SEED)
    optimizer = torch.optim.AdamW(_slow_parameters(model), lr=scratch.LR, weight_decay=0.01)
    data = scratch._tensors(train, device)
    best_far = -1
    best_state = None
    for epoch in range(1, scratch.EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(train), scratch.BATCH):
            batch = data[order[start : start + scratch.BATCH]]
            forged = scratch._corrupt(batch, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, forged)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is not finite at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_slow_parameters(model), 1.0)
            optimizer.step()
            model.reset_fast()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        if epoch == 1 or epoch % 10 == 0 or epoch == scratch.EPOCHS:
            model.eval()
            model.reset_fast()
            train_hits, train_n = _exact(model, scratch._span(train), device)
            far_hits, far_n = _exact(model, banks["far"], device)
            mul_rows = [row for row in banks["far"] if row[0] == scratch.OP_MUL]
            mul_hits, mul_n = _exact(model, mul_rows, device)
            split = _add_split(model, banks["far"], device)
            print(
                f"epoch {epoch} loss {total / seen:.4f} train {train_hits}/{train_n} far {far_hits}/{far_n} mul {mul_hits}/{mul_n} {split} gpu_mib {torch.cuda.memory_allocated() / 1024**2:.1f}",
                flush=True,
            )
            if far_hits > best_far:
                best_far = far_hits
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no epoch was scored")
    model.load_state_dict(best_state)
    print(f"kept far {best_far}/{len(banks['far'])}", flush=True)


def _one_hot_value(value: int, device: torch.device) -> torch.Tensor:
    contrib = torch.zeros(1, NCONTRIB, device=device)
    contrib[0, value - VMIN] = 1.0
    return contrib


def _offset_ok(model: Place, device: torch.device) -> None:
    """The carry is an index shift, and a later digit loss can still move it."""
    if model.sweep1.in_features != 4 * H or model.read.out_features != NCONTRIB:
        raise RuntimeError("sweep still sees the carry or the contribution width drifted")

    def combine(value: int, carry: int, direction: int) -> tuple[int, int]:
        carried = torch.zeros(1, CARRY, device=device)
        carried[0, carry] = 1.0
        way = torch.tensor([direction], dtype=torch.long, device=device)
        digit_log, nxt = model._combine(_one_hot_value(value, device), carried, way)
        return int(digit_log.argmax(dim=-1)), int(nxt.argmax(dim=-1))

    checks = {
        (12, 0, 1): (2, 1),
        (6, 1, 1): (7, 0),
        (-2, 0, -1): (8, 1),
        (6, 1, 0): (6, 0),
    }
    for key, expect in checks.items():
        got = combine(*key)
        if got != expect:
            raise RuntimeError(f"offset {key} -> {got}, expected {expect}")

    zero = F.one_hot(torch.zeros(1, dtype=torch.long, device=device), CARRY).to(torch.float32)
    way = torch.ones(1, dtype=torch.long, device=device)
    digit0, mid = model._combine(_one_hot_value(12, device), zero, way)
    digit1, end = model._combine(_one_hot_value(6, device), mid, way)
    chain = (int(digit0.argmax()), int(digit1.argmax()), int(end.argmax()))
    if chain != (2, 7, 0):
        raise RuntimeError(f"chain 12 then 6 -> {chain}")

    leaf = torch.full((1, NCONTRIB), -50.0, device=device)
    leaf[0, 2 - VMIN] = 0.0
    leaf[0, 12 - VMIN] = 0.0
    leaf.requires_grad_(True)
    fixed = _one_hot_value(6, device)
    _digit0, mid = model._combine(torch.softmax(leaf, dim=-1), zero, way)
    digit1, _end = model._combine(fixed, mid, way)
    (-digit1[0, 7]).backward()
    grad = leaf.grad
    prefer = grad is not None and float(grad[0, 12 - VMIN]) < 0.0 and float(grad[0, 12 - VMIN]) < float(grad[0, 2 - VMIN])
    if not prefer:
        raise RuntimeError("tens digit loss does not prefer contribution 12 over 2")
    model.zero_grad(set_to_none=True)
    print("offset carry ok", flush=True)


def _one_hot_width(value: int, width: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(1, width, device=device)
    out[0, value] = 1.0
    return out


def _product_ok(model: Place, device: torch.device) -> None:
    """A product carry is an offset, and the next digit can still prefer 20 over 10."""
    if model.product.in_features != H or model.product.out_features != NPROD:
        raise RuntimeError("product head width drifted")
    if model.mul_sign_carry.in_features != MUL_CARRY or model.mul_sign_carry.out_features != H:
        raise RuntimeError("product carry width drifted")
    zero = _one_hot_width(0, MUL_CARRY, device)
    prod20 = _one_hot_width(20, NPROD, device)
    prod0 = _one_hot_width(0, NPROD, device)
    digit0, carry0 = model._mul_digit(_convolve(prod20, zero))
    digit1, carry1 = model._mul_digit(_convolve(prod0, carry0))
    chain = (int(digit0.argmax()), int(carry0.argmax()), int(digit1.argmax()), int(carry1.argmax()))
    if chain != (0, 2, 2, 0):
        raise RuntimeError(f"product 20 then 0 -> {chain}")
    paired = _convolve(_convolve(prod20, _one_hot_width(6, NPROD, device)), zero)
    digit, carry = model._mul_digit(paired)
    if (int(digit.argmax()), int(carry.argmax())) != (6, 2):
        raise RuntimeError(f"products 20 and 6 -> {(int(digit.argmax()), int(carry.argmax()))}")
    eighty_one = _one_hot_width(81, NPROD, device)
    total = eighty_one
    for _ in range(3):
        total = _convolve(total, eighty_one)
    total = _convolve(total, zero)
    if abs(float(total.sum()) - 1.0) > 1e-4:
        raise RuntimeError(f"four products do not sum to 1 ({float(total.sum())})")
    digit, carry = model._mul_digit(total)
    if (int(digit.argmax()), int(carry.argmax())) != (4, 32):
        raise RuntimeError(f"four 81s -> {(int(digit.argmax()), int(carry.argmax()))}")
    leaf = torch.full((1, NPROD), -50.0, device=device)
    leaf[0, 10] = 0.0
    leaf[0, 20] = 0.0
    leaf.requires_grad_(True)
    _digit0, mid = model._mul_digit(_convolve(torch.softmax(leaf, dim=-1), zero))
    digit1, _end = model._mul_digit(_convolve(prod0, mid))
    (-digit1[0, 2]).backward()
    grad = leaf.grad
    prefer = grad is not None and float(grad[0, 20]) < 0.0 and float(grad[0, 20]) < float(grad[0, 10])
    if not prefer:
        raise RuntimeError("next-place digit loss does not prefer product 20 over 10")

    learned = torch.zeros(1, PLACES, PLACES, NPROD, device=device)
    learned[..., 80] = 1
    learned[0, 0, 0, :] = 0
    learned[0, 0, 0, 20] = 1
    left = torch.tensor([[4, 0, 0, 0]], device=device)
    right = torch.tensor([[5, 0, 0, 0]], device=device)
    gated = model._gate(learned, left, right)
    if int(gated[0, 0, 0].argmax()) != 20 or float(gated[0, 0, 1, 0]) != 1.0 or float(gated[0, 1, 1, 0]) != 1.0:
        raise RuntimeError("zero digit did not contribute product 0")
    fired = torch.zeros(1, PLACES, PLACES, H, device=device)
    digit_log, _hidden, end_carry = model._from_pairs(gated, fired)
    written = tuple(int(item) for item in digit_log.argmax(dim=-1)[0])
    if written != (0, 2, 0, 0) or int(end_carry.argmax()) != 0:
        raise RuntimeError(f"gated 4*5 -> {written} carry {int(end_carry.argmax())}")
    logits = torch.full((1, PLACES, PLACES, NPROD), -50.0, device=device)
    logits[..., 80] = 0.0
    logits[0, 0, 0, :] = -50.0
    logits[0, 0, 0, 10] = 0.0
    logits[0, 0, 0, 20] = 0.0
    logits.requires_grad_(True)
    digit_log, _hidden, _end = model._from_pairs(model._gate(torch.softmax(logits, dim=-1), left, right), fired)
    (-digit_log[0, 1, 2]).backward()
    choice = logits.grad
    if choice is None or not (float(choice[0, 0, 0, 20]) < 0.0 and float(choice[0, 0, 0, 20]) < float(choice[0, 0, 0, 10])):
        raise RuntimeError("gated next place does not prefer product 20 over 10")
    if float(choice[0, 0, 1].abs().sum()) != 0.0:
        raise RuntimeError("a zero pair still receives a product gradient")

    def magnitude(param: nn.Parameter) -> float:
        if param.grad is None:
            return 0.0
        return float(param.grad.detach().abs().sum())

    def route(op: int) -> dict[str, float]:
        model.zero_grad(set_to_none=True)
        _sign, digits, _hidden = model.core(*model._read_prompt(_prompt(op, 4, 5, device)))
        digits.sum().backward()
        named = {
            "product": model.product.weight,
            "read": model.read.weight,
            "digit": model.digit.weight,
            "mix": model.mix1.weight,
            "prod_digit": model.prod_digit.weight,
            "prod_mix": model.prod_mix1.weight,
        }
        return {name: magnitude(param) for name, param in named.items()}

    mul = route(scratch.OP_MUL)
    add = route(scratch.OP_ADD)
    div = route(scratch.OP_DIV)
    if mul["product"] == 0.0 or mul["prod_digit"] == 0.0 or mul["prod_mix"] == 0.0 or mul["read"] != 0.0:
        raise RuntimeError(f"multiplication still uses the sweep ({mul})")
    if mul["digit"] != 0.0 or mul["mix"] != 0.0:
        raise RuntimeError(f"multiplication reaches the addition mix ({mul})")
    if add["product"] != 0.0 or add["prod_digit"] != 0.0 or add["prod_mix"] != 0.0 or add["read"] == 0.0 or add["digit"] == 0.0:
        raise RuntimeError(f"addition reaches the product mix ({add})")
    if div["product"] != 0.0 or div["prod_digit"] != 0.0 or div["read"] == 0.0:
        raise RuntimeError(f"division reaches the product mix ({div})")
    model.zero_grad(set_to_none=True)
    print("product carry ok", flush=True)


def _add_split(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> str:
    adds = [row for row in rows if row[0] == scratch.OP_ADD]
    plain = [row for row in adds if (row[1] % 10) + (row[2] % 10) < 10]
    carried = [row for row in adds if (row[1] % 10) + (row[2] % 10) >= 10]
    plain_hits, plain_n = _exact(model, plain, device)
    carry_hits, carry_n = _exact(model, carried, device)
    return f"add_plain {plain_hits}/{plain_n} add_carry {carry_hits}/{carry_n}"


@torch.no_grad()
def _add_places(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> str:
    adds = [row for row in rows if row[0] == scratch.OP_ADD]
    ones = tens = 0
    for start in range(0, len(adds), scratch.BATCH):
        chunk = adds[start : start + scratch.BATCH]
        prompt = scratch._tensors(chunk, device)[:, :PROMPT_LEN]
        drafted, _sign, _digits, _hidden = model.predict_bundle(prompt)
        for row, draft in zip(chunk, drafted.tolist()):
            truth = scratch._number(row[3])
            if draft[1] == truth[1]:
                ones += 1
            if draft[2] == truth[2]:
                tens += 1
    return f"ones {ones}/{len(adds)} tens {tens}/{len(adds)}"


def _probes(model: Place, device: torch.device) -> None:
    adds = ((8, 7), (5, 7), (3, 3), (30, 30), (31, 31), (35, 37), (40, 40), (45, 45))
    for left, right in adds:
        drafted, pred, _called, _yes, _digits = _one(model, scratch.OP_ADD, left, right, device)
        truth = scratch._truth(scratch.OP_ADD, left, right)
        print(f"probe {left}+{right} truth {truth} pred {pred} tokens {drafted}", flush=True)
    products = ((4, 5), (6, 7), (9, 9), (12, 12), (31, 32), (31, 35), (40, 31), (45, 45))
    for left, right in products:
        drafted, pred, _called, _yes, _digits = _one(model, scratch.OP_MUL, left, right, device)
        truth = scratch._truth(scratch.OP_MUL, left, right)
        print(f"probe {left}*{right} truth {truth} pred {pred} tokens {drafted}", flush=True)


def _smoke(model: Place, banks: dict[str, list[tuple[int, int, int, int]]], device: torch.device) -> None:
    _offset_ok(model, device)
    _product_ok(model, device)
    sample = scratch._tensors(banks["train"][:4], device)
    sign_logits, digit_logits, hidden = model.core(*model._read_prompt(sample[:, :PROMPT_LEN]))
    if sign_logits.shape != (4, 2) or digit_logits.shape != (4, PLACES, BASE) or hidden.shape != (4, PLACES, H):
        raise RuntimeError(f"shapes {tuple(sign_logits.shape)} {tuple(digit_logits.shape)} {tuple(hidden.shape)}")
    (sign_logits.sum() + digit_logits.sum()).backward()
    model.zero_grad(set_to_none=True)
    model.reset_fast()
    if float(model.fast_up.detach().norm()) != 0.0:
        raise RuntimeError("short-term layer is not silent")
    print(f"place forward ok shared-mix {tuple(model.mix1.weight.shape)}", flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    banks = scratch._banks()
    scratch._audit(banks)
    torch.manual_seed(scratch.SEED)
    torch.cuda.manual_seed_all(scratch.SEED)
    device = torch.device("cuda")
    model = Place().to(device)
    if any(param.device.type != "cuda" for param in model.parameters()):
        raise SystemExit("a parameter is off the GPU")
    print(
        f"gpu_mib {torch.cuda.memory_allocated() / 1024**2:.1f} params {sum(param.numel() for param in model.parameters())}",
        flush=True,
    )
    _smoke(model, banks, device)
    _fit(model, banks["train"], banks, device)
    model.eval()
    model.reset_fast()
    report = {}
    for name in ("train", "in", "far", "stiff"):
        sample = banks[name] if name != "train" else scratch._span(banks["train"])
        hits, count = _exact(model, sample, device)
        copied = scratch._copy_left(sample)
        verdict_hits, verdict_n = _verdict_hits(model, sample, device)
        repaired, repaired_n = _verdict_repair_hits(model, sample, device)
        report[name] = {
            "exact": hits,
            "count": count,
            "copy_left": copied,
            "verdict": verdict_hits,
            "verdict_n": verdict_n,
            "verdict_repair": repaired,
            "verdict_repair_n": repaired_n,
        }
        print(
            f"{name} {hits}/{count} copy_left {copied}/{count} verdict {verdict_hits}/{verdict_n} verdict_repair {repaired}/{repaired_n}",
            flush=True,
        )
    for name in ("in", "far", "stiff"):
        print(f"{name} ops {_by_op(model, banks[name], device)}", flush=True)
    print(f"far {_add_split(model, banks['far'], device)} {_add_places(model, banks['far'], device)}", flush=True)
    _probes(model, device)
    for op, left, right, value in banks["stiff"]:
        drafted, pred, _called, _yes, _digits = _one(model, op, left, right, device)
        print(f"stiff {(op, left, right, value)} pred {pred} tokens {drafted}", flush=True)
    rejected, changed, improved = _revise(model, banks["stiff"] + banks["far"][:64], device)
    print(f"revise rejected {rejected} changed {changed} improved {improved}", flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEST / "model.pt")
    (DEST / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(DEST / "model.pt", DEST / "factor.pt")
    shutil.copy2(DEST / "scores.json", DEST / "factor_scores.json")
    print(f"wrote {DEST / 'model.pt'}", flush=True)
    print(f"wrote {DEST / 'factor.pt'}", flush=True)


def _combine_many(
    model: Place,
    lefts: list[int],
    rights: list[int],
    device: torch.device,
    direction: int,
) -> tuple[list[int], list[int]]:
    """One place-value step. Direction +1 adds and -1 subtracts. No learned weight is read."""
    if direction not in (1, -1):
        raise RuntimeError(f"direction {direction}")
    if len(lefts) != len(rights) or not lefts:
        raise RuntimeError("place step is empty or uneven")
    batch = len(lefts)
    contrib = torch.zeros(batch, NCONTRIB, device=device)
    carry = torch.zeros(batch, CARRY, device=device)
    for index, (left, right) in enumerate(zip(lefts, rights)):
        if not VMIN <= left <= VMAX:
            raise RuntimeError(f"value {left} is outside the offset")
        if not 0 <= right < CARRY:
            raise RuntimeError(f"carry {right} is outside the offset")
        contrib[index, left - VMIN] = 1.0
        carry[index, right] = 1.0
    way = torch.full((batch,), direction, dtype=torch.long, device=device)
    digit_log, nxt = model._combine(contrib, carry, way)
    return digit_log.argmax(dim=-1).tolist(), nxt.argmax(dim=-1).tolist()


def _combine_pair(
    model: Place, left: int, right: int, device: torch.device, direction: int = 1
) -> tuple[int, int]:
    digits, nxts = _combine_many(model, [left], [right], device, direction)
    return digits[0], nxts[0]


def _operand_digits(value: int) -> list[int]:
    tokens = scratch._number(value)
    return [token - DIGIT0 for token in tokens[1:]]


def _digits_of(tokens: list[int]) -> list[int]:
    """Places of an answer the chain itself wrote. The integer is not kept."""
    if scratch._read_number(tokens) is None or tokens[0] != PLUS_SIGN:
        raise RuntimeError(f"chain answer does not read: {tokens}")
    return [token - DIGIT0 for token in tokens[1:]]


def _answer(digits: list[int]) -> list[int]:
    tokens = [PLUS_SIGN, *[DIGIT0 + digit for digit in digits]]
    if scratch._read_number(tokens) is None:
        raise RuntimeError(f"chain digits do not write: {digits}")
    return tokens


def _equation(left: list[int], right: list[int], total: list[int], op: int) -> list[int]:
    row = left + [op] + right + [scratch.EQ] + total
    if len(row) != scratch.LENGTH or row[scratch.PROMPT_LEN :] != total:
        raise RuntimeError("chain equation drifted")
    return row


def _is_zero(tokens: list[int]) -> bool:
    return all(token == DIGIT0 for token in tokens[1:])


@torch.no_grad()
def _rule_add_many(model: Place, lefts: list[list[int]], rights: list[list[int]], device: torch.device) -> list[list[int]]:
    """Add written answers one place at a time. The next step reads these tokens."""
    left_digits = [_digits_of(tokens) for tokens in lefts]
    right_digits = [_digits_of(tokens) for tokens in rights]
    carry = [0] * len(lefts)
    columns = [[] for _ in lefts]
    for place in range(PLACES):
        digit_ab, carry_ab = _combine_many(
            model,
            [row[place] for row in left_digits],
            [row[place] for row in right_digits],
            device,
            1,
        )
        digit, carry_extra = _combine_many(model, digit_ab, carry, device, 1)
        carry, overflow = _combine_many(model, carry_ab, carry_extra, device, 1)
        if any(item != 0 for item in overflow):
            raise RuntimeError("a place carry did not fit")
        for index, item in enumerate(digit):
            columns[index].append(item)
    if any(item != 0 for item in carry):
        raise RuntimeError("the sum does not fit in four places")
    totals = [_answer(column) for column in columns]
    for left, right, total in zip(lefts, rights, totals):
        _equation(left, right, total, scratch.OP_ADD)
    return totals


def _rule_add(model: Place, left: list[int], right: list[int], device: torch.device) -> list[int]:
    return _rule_add_many(model, [left], [right], device)[0]


def _positive_magnitude(model: Place, rows: list[list[int]], device: torch.device) -> list[list[int]]:
    """Magnitude of a borrowed difference: nine's complement, then add one."""
    flipped = [[] for _ in rows]
    for place in range(PLACES):
        digit, borrow = _combine_many(model, [9] * len(rows), [row[place] for row in rows], device, -1)
        if any(item != 0 for item in borrow):
            raise RuntimeError("a nine's complement borrowed")
        for index, item in enumerate(digit):
            flipped[index].append(item)
    return _rule_add_many(
        model,
        [_answer(row) for row in flipped],
        [scratch._number(1) for _ in rows],
        device,
    )


@torch.no_grad()
def _rule_sub_many(model: Place, lefts: list[list[int]], rights: list[list[int]], device: torch.device) -> list[list[int]]:
    """Subtract by the offset. A final borrow of 1 is a minus sign on the complement."""
    left_digits = [_digits_of(tokens) for tokens in lefts]
    right_digits = [_digits_of(tokens) for tokens in rights]
    borrow = [0] * len(lefts)
    columns = [[] for _ in lefts]
    for place in range(PLACES):
        digit_ab, borrow_ab = _combine_many(
            model,
            [row[place] for row in left_digits],
            [row[place] for row in right_digits],
            device,
            -1,
        )
        digit, borrow_extra = _combine_many(model, digit_ab, borrow, device, -1)
        borrow, overflow = _combine_many(model, borrow_ab, borrow_extra, device, 1)
        if any(item != 0 for item in overflow) or any(item not in (0, 1) for item in borrow):
            raise RuntimeError("a place borrow did not fit")
        for index, item in enumerate(digit):
            columns[index].append(item)
    totals: list[list[int] | None] = [None] * len(lefts)
    borrowed: list[list[int]] = []
    borrowed_at: list[int] = []
    for index, column in enumerate(columns):
        if borrow[index] == 0:
            totals[index] = _answer(column)
        else:
            borrowed.append(column)
            borrowed_at.append(index)
    if borrowed:
        magnitudes = _positive_magnitude(model, borrowed, device)
        for index, magnitude in zip(borrowed_at, magnitudes):
            signed = [MINUS_SIGN, *magnitude[1:]]
            if scratch._read_number(signed) is None:
                raise RuntimeError(f"negative difference does not read: {signed}")
            totals[index] = signed
    written = [tokens for tokens in totals if tokens is not None]
    if len(written) != len(lefts):
        raise RuntimeError("a difference was not written")
    for left, right, total in zip(lefts, rights, written):
        _equation(left, right, total, scratch.OP_SUB)
    return written


def _rule_sub(model: Place, left: list[int], right: list[int], device: torch.device) -> list[int]:
    return _rule_sub_many(model, [left], [right], device)[0]


@torch.no_grad()
def _rule_div_many(
    model: Place,
    dividends: list[list[int]],
    divisors: list[list[int]],
    device: torch.device,
) -> tuple[list[list[int]], list[list[int]]]:
    """Quotient is how many times the divisor subtracts before the sign would change."""
    if any(_is_zero(tokens) for tokens in divisors):
        raise RuntimeError("divide by zero")
    remainders = list(dividends)
    counts = [scratch._number(0) for _ in dividends]
    active = [True] * len(dividends)
    for _ in range(BASE**PLACES):
        if not any(active):
            break
        trials = _rule_sub_many(model, remainders, divisors, device)
        still: list[int] = []
        for index, trial in enumerate(trials):
            if not active[index] or trial[0] == MINUS_SIGN:
                active[index] = False
                continue
            remainders[index] = trial
            still.append(index)
        if not still:
            break
        updated = _rule_add_many(
            model,
            [counts[index] for index in still],
            [scratch._number(1) for _ in still],
            device,
        )
        for index, count in zip(still, updated):
            counts[index] = count
    if any(active):
        raise RuntimeError("division did not stop")
    return counts, remainders


def _rule_div(model: Place, dividend: list[int], divisor: list[int], device: torch.device) -> tuple[list[int], list[int]]:
    counts, remainders = _rule_div_many(model, [dividend], [divisor], device)
    return counts[0], remainders[0]


def _shift_places(digits: list[int], places: int) -> list[int]:
    """Move a written number up by places. A shift is the place-value rule."""
    out = [0] * PLACES
    for index, digit in enumerate(digits):
        if digit == 0:
            continue
        dest = index + places
        if dest >= PLACES:
            raise RuntimeError(f"shift of {digits} by {places} does not fit")
        out[dest] = digit
    return out


@torch.no_grad()
def _repeat_add(model: Place, digit: int, count: int, device: torch.device) -> tuple[list[int], list[str]]:
    """Add one digit, count times. The next line reads the answer tokens just written."""
    total = scratch._number(0)
    addend = scratch._number(digit)
    lines = []
    for _ in range(count):
        before = total
        total = _rule_add(model, before, addend, device)
        _equation(before, addend, total, scratch.OP_ADD)
        lines.append(f"{scratch._read_number(before)}+{digit}={scratch._read_number(total)}")
    return total, lines


@torch.no_grad()
def _think_mul(model: Place, left: int, right: int, device: torch.device, speak: bool) -> list[int]:
    """Break a product into digit chains. Partials and the running total are answer tokens."""
    if speak:
        print(f"think {left}*{right}", flush=True)
    partials = []
    for index, left_digit in enumerate(_operand_digits(left)):
        for jndex, right_digit in enumerate(_operand_digits(right)):
            if left_digit == 0 or right_digit == 0:
                continue
            product, lines = _repeat_add(model, left_digit, right_digit, device)
            shifted = _answer(_shift_places(_digits_of(product), index + jndex))
            partials.append(shifted)
            if speak:
                print(f"  digit {left_digit}*{right_digit}", flush=True)
                for line in lines:
                    print(f"  {line}", flush=True)
                print(f"  place {index}+{jndex} -> {scratch._read_number(shifted)}", flush=True)
    total = scratch._number(0)
    for partial in partials:
        before = total
        total = _rule_add(model, before, partial, device)
        _equation(before, partial, total, scratch.OP_ADD)
        if speak:
            print(
                f"  {scratch._read_number(before)}+{scratch._read_number(partial)}={scratch._read_number(total)}",
                flush=True,
            )
    return total


def _chain_answer(model: Place, op: int, left: int, right: int, device: torch.device) -> list[int] | None:
    """One answer from the place-value chain. A negative operand stays on the learned pass."""
    if left < 0 or right < 0:
        return None
    left_tokens = scratch._number(left)
    right_tokens = scratch._number(right)
    if op == scratch.OP_ADD:
        return _rule_add(model, left_tokens, right_tokens, device)
    if op == scratch.OP_SUB:
        return _rule_sub(model, left_tokens, right_tokens, device)
    if op == scratch.OP_MUL:
        return _think_mul(model, left, right, device, speak=False)
    if op == scratch.OP_DIV:
        if right == 0:
            return None
        quotient, remainder = _rule_div(model, left_tokens, right_tokens, device)
        if not _is_zero(remainder):
            return None
        return quotient
    if op == scratch.OP_LARGER:
        diff = _rule_sub(model, left_tokens, right_tokens, device)
        return right_tokens if diff[0] == MINUS_SIGN else left_tokens
    if op == scratch.OP_SMALLER:
        diff = _rule_sub(model, left_tokens, right_tokens, device)
        return left_tokens if diff[0] == MINUS_SIGN or _is_zero(diff) else right_tokens
    return None


def _require(kind: str, left: int, right: int, got: int | None, expect: int) -> None:
    print(f"rule {left}{kind}{right}={got}", flush=True)
    if got != expect:
        raise RuntimeError(f"rule {left}{kind}{right}={got}")


def _report_think() -> None:
    """Score the offset chain on the far bank, then use each product on a harder check."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    device = torch.device("cuda")
    model = Place().to(device)
    path = DEST / "factor.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    for left, right, expect in ((4, 5, 9), (8, 1, 9), (49, 0, 49), (7, 7, 14)):
        got = scratch._read_number(_rule_add(model, scratch._number(left), scratch._number(right), device))
        _require("+", left, right, got, expect)
    for left, right, expect in ((7, 3, 4), (3, 7, -4), (31, 45, -14), (45, 31, 14), (40, 40, 0)):
        got = scratch._read_number(_rule_sub(model, scratch._number(left), scratch._number(right), device))
        _require("-", left, right, got, expect)
    for left, right, expect in ((3, 9, 9), (9, 3, 9), (5, 5, 5)):
        diff = _rule_sub(model, scratch._number(left), scratch._number(right), device)
        larger = scratch._number(right) if diff[0] == MINUS_SIGN else scratch._number(left)
        _require(" larger ", left, right, scratch._read_number(larger), expect)
    for left, right, expect in ((9, 3, 3), (3, 9, 3), (5, 5, 5)):
        diff = _rule_sub(model, scratch._number(left), scratch._number(right), device)
        negative = diff[0] == MINUS_SIGN
        zero = _is_zero(diff)
        smaller = scratch._number(left) if negative or zero else scratch._number(right)
        _require(" smaller ", left, right, scratch._read_number(smaller), expect)
    for dividend, divisor, expect, remain in ((14, 7, 2, 0), (14, 3, 4, 2)):
        quot, rem = _rule_div(model, scratch._number(dividend), scratch._number(divisor), device)
        got = scratch._read_number(quot)
        left_over = scratch._read_number(rem)
        print(f"rule {dividend}/{divisor}={got} rem {left_over}", flush=True)
        if got != expect or left_over != remain:
            raise RuntimeError(f"rule {dividend}/{divisor}={got} rem {left_over}")
    for left, right in ((7, 7), (37, 37)):
        product = _think_mul(model, left, right, device, speak=True)
        used = _rule_add(model, product, scratch._number(right), device)
        one = _rule_add(model, product, scratch._number(1), device)
        quot, rem = _rule_div(model, product, scratch._number(right), device)
        print(f"use {scratch._read_number(product)}+{right}={scratch._read_number(used)}", flush=True)
        print(f"use {scratch._read_number(product)}+1={scratch._read_number(one)}", flush=True)
        print(
            f"use {scratch._read_number(product)}/{right}={scratch._read_number(quot)} rem {scratch._read_number(rem)}",
            flush=True,
        )
    banks = scratch._banks()
    sub_rows = [row for row in banks["far"] if row[0] == scratch.OP_SUB]
    larger_rows = [row for row in banks["far"] if row[0] == scratch.OP_LARGER]
    smaller_rows = [row for row in banks["far"] if row[0] == scratch.OP_SMALLER]
    div_rows = [row for row in banks["far"] if row[0] == scratch.OP_DIV]
    mul_rows = [row for row in banks["far"] if row[0] == scratch.OP_MUL]
    differences = _rule_sub_many(
        model,
        [scratch._number(left) for _op, left, _right, _value in sub_rows],
        [scratch._number(right) for _op, _left, right, _value in sub_rows],
        device,
    )
    sub_hits = 0
    for (_op, left, right, value), diff in zip(sub_rows, differences):
        got = scratch._read_number(diff)
        if got == value:
            sub_hits += 1
        else:
            print(f"miss {left}-{right} truth {value} chain {got}", flush=True)
    print(f"chain far sub {sub_hits}/{len(sub_rows)}", flush=True)
    pairs = [(left, right) for _op, left, right, _value in sub_rows]
    if pairs != [(left, right) for _op, left, right, _value in larger_rows]:
        raise RuntimeError("far larger pairs do not follow subtraction")
    if pairs != [(left, right) for _op, left, right, _value in smaller_rows]:
        raise RuntimeError("far smaller pairs do not follow subtraction")
    larger_hits = 0
    for (_op, left, right, value), diff in zip(larger_rows, differences):
        choice = scratch._number(right) if diff[0] == MINUS_SIGN else scratch._number(left)
        got = scratch._read_number(choice)
        if got == value:
            larger_hits += 1
        else:
            print(f"miss larger {left} {right} truth {value} chain {got}", flush=True)
    print(f"chain far larger {larger_hits}/{len(larger_rows)}", flush=True)
    smaller_hits = 0
    for (_op, left, right, value), diff in zip(smaller_rows, differences):
        negative = diff[0] == MINUS_SIGN
        choice = scratch._number(left) if negative or _is_zero(diff) else scratch._number(right)
        got = scratch._read_number(choice)
        if got == value:
            smaller_hits += 1
        else:
            print(f"miss smaller {left} {right} truth {value} chain {got}", flush=True)
    print(f"chain far smaller {smaller_hits}/{len(smaller_rows)}", flush=True)
    quotients, remainders = _rule_div_many(
        model,
        [scratch._number(left) for _op, left, _right, _value in div_rows],
        [scratch._number(right) for _op, _left, right, _value in div_rows],
        device,
    )
    div_hits = 0
    for (_op, left, right, value), quot, rem in zip(div_rows, quotients, remainders):
        got = scratch._read_number(quot)
        if got == value and _is_zero(rem):
            div_hits += 1
        else:
            print(f"miss {left}/{right} truth {value} chain {got} rem {scratch._read_number(rem)}", flush=True)
    print(f"chain far div {div_hits}/{len(div_rows)}", flush=True)
    hits = 0
    plus_hits = 0
    other_hits = 0
    products: list[list[int]] = []
    for index, (_op, left, right, value) in enumerate(mul_rows, start=1):
        product = _think_mul(model, left, right, device, speak=False)
        products.append(product)
        got = scratch._read_number(product)
        if got == value:
            hits += 1
        else:
            print(f"miss {left}*{right} truth {value} chain {got}", flush=True)
        plus = scratch._read_number(_rule_add(model, product, scratch._number(1), device))
        if plus == value + 1:
            plus_hits += 1
        else:
            print(f"miss {left}*{right}+1 truth {value + 1} chain {plus}", flush=True)
        other = scratch._read_number(_rule_add(model, product, scratch._number(right), device))
        if other == value + right:
            other_hits += 1
        else:
            print(f"miss {left}*{right}+{right} truth {value + right} chain {other}", flush=True)
        if index % 25 == 0:
            print(f"scored mul {index}/{len(mul_rows)}", flush=True)
    print(f"chain far mul {hits}/{len(mul_rows)}", flush=True)
    print(f"chain product+1 {plus_hits}/{len(mul_rows)}", flush=True)
    print(f"chain product+right {other_hits}/{len(mul_rows)}", flush=True)
    by_right, right_rem = _rule_div_many(
        model,
        products,
        [scratch._number(right) for _op, _left, right, _value in mul_rows],
        device,
    )
    by_left, left_rem = _rule_div_many(
        model,
        products,
        [scratch._number(left) for _op, left, _right, _value in mul_rows],
        device,
    )
    right_hits = 0
    left_hits = 0
    for (_op, left, right, _value), quot, rem, product in zip(mul_rows, by_right, right_rem, products):
        got = scratch._read_number(quot)
        if got == left and _is_zero(rem):
            right_hits += 1
        else:
            print(
                f"miss {scratch._read_number(product)}/{right} chain {got} rem {scratch._read_number(rem)}",
                flush=True,
            )
    for (_op, left, right, _value), quot, rem, product in zip(mul_rows, by_left, left_rem, products):
        got = scratch._read_number(quot)
        if got == right and _is_zero(rem):
            left_hits += 1
        else:
            print(
                f"miss {scratch._read_number(product)}/{left} chain {got} rem {scratch._read_number(rem)}",
                flush=True,
            )
    print(f"chain product/right {right_hits}/{len(mul_rows)}", flush=True)
    print(f"chain product/left {left_hits}/{len(mul_rows)}", flush=True)
    print(f"read {path}", flush=True)


def _sweep_parameters(model: Place) -> list[nn.Parameter]:
    """The place sweep. The product head is not part of this lesson."""
    prefixes = ("digit.", "sign.", "op.", "mix1.", "mix2.", "sweep1.", "sweep2.", "read.", "sign_carry.", "sign_read.")
    chosen = [(name, param) for name, param in model.named_parameters() if name.startswith(prefixes)]
    expected = {
        "digit.weight",
        "sign.weight",
        "op.weight",
        "mix1.weight",
        "mix1.bias",
        "mix2.weight",
        "mix2.bias",
        "sweep1.weight",
        "sweep1.bias",
        "sweep2.weight",
        "sweep2.bias",
        "read.weight",
        "read.bias",
        "sign_carry.weight",
        "sign_carry.bias",
        "sign_read.weight",
        "sign_read.bias",
    }
    if {name for name, _param in chosen} != expected:
        raise RuntimeError(f"sweep parameters {[name for name, _param in chosen]}")
    return [param for _name, param in chosen]


def _offset_value(digit: int, step: int, direction: int) -> int:
    """The place total whose offset is this digit and carry. The encoding is the one _combine uses."""
    value = digit + BASE * step if direction == 1 else digit - BASE * step
    if not VMIN <= value <= VMAX:
        raise RuntimeError(f"offset value {value} is outside the sweep")
    return value


@torch.no_grad()
def _check_offset_bin(model: Place, device: torch.device) -> None:
    def recovered(value: int, direction: int) -> tuple[int, int]:
        contrib = torch.zeros(1, NCONTRIB, device=device)
        contrib[0, value - VMIN] = 1.0
        carry = torch.zeros(1, CARRY, device=device)
        carry[0, 0] = 1.0
        way = torch.tensor([direction], dtype=torch.long, device=device)
        digit, nxt = model._combine(contrib, carry, way)
        return int(digit.argmax()), int(nxt.argmax())

    pairs = {1: ((4, 5), (9, 9), (0, 0), (8, 1)), -1: ((7, 3), (3, 7), (5, 5), (0, 1))}
    for direction, samples in pairs.items():
        for left, right in samples:
            digit, step = _combine_pair(model, left, right, device, direction)
            value = _offset_value(digit, step, direction)
            if recovered(value, direction) != (digit, step):
                raise RuntimeError(f"offset bin {left} {right} direction {direction} value {value}")


def _place_bins(model: Place, lefts: list[list[int]], rights: list[list[int]], direction: int, device: torch.device) -> list[list[int]]:
    columns = [[] for _ in lefts]
    for place in range(PLACES):
        digits, steps = _combine_many(
            model,
            [tokens[1 + place] - DIGIT0 for tokens in lefts],
            [tokens[1 + place] - DIGIT0 for tokens in rights],
            device,
            direction,
        )
        for index, (digit, step) in enumerate(zip(digits, steps)):
            columns[index].append(_offset_value(digit, step, direction) - VMIN)
    return columns


@torch.no_grad()
def _chain_lessons(
    model: Place, rows: list[tuple[int, int, int, int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train-bank lessons from the chain. A negative operand is not in this lesson."""
    grouped: dict[int, list[int]] = {op: [] for op in scratch.OPS}
    for index, (op, left, right, _value) in enumerate(rows):
        if op == scratch.OP_MUL or left < 0 or right < 0:
            continue
        grouped[op].append(index)
    answers: list[list[int] | None] = [None] * len(rows)
    bins = [[-1] * PLACES for _ in rows]
    for op, indexes in grouped.items():
        if not indexes:
            continue
        lefts = [scratch._number(rows[index][1]) for index in indexes]
        rights = [scratch._number(rows[index][2]) for index in indexes]
        if op == scratch.OP_ADD:
            totals = _rule_add_many(model, lefts, rights, device)
            for index, row_bins in zip(indexes, _place_bins(model, lefts, rights, 1, device)):
                bins[index] = row_bins
        elif op == scratch.OP_SUB:
            totals = _rule_sub_many(model, lefts, rights, device)
            for index, row_bins in zip(indexes, _place_bins(model, lefts, rights, -1, device)):
                bins[index] = row_bins
        elif op == scratch.OP_DIV:
            totals, remainders = _rule_div_many(model, lefts, rights, device)
            for index, rem in zip(indexes, remainders):
                if not _is_zero(rem):
                    raise RuntimeError("a training division left a remainder")
        elif op == scratch.OP_LARGER:
            diffs = _rule_sub_many(model, lefts, rights, device)
            totals = [right if diff[0] == MINUS_SIGN else left for diff, left, right in zip(diffs, lefts, rights)]
        elif op == scratch.OP_SMALLER:
            diffs = _rule_sub_many(model, lefts, rights, device)
            totals = [
                left if diff[0] == MINUS_SIGN or _is_zero(diff) else right
                for diff, left, right in zip(diffs, lefts, rights)
            ]
        else:
            continue
        for index, total in zip(indexes, totals):
            got = scratch._read_number(total)
            _op, left, right, value = rows[index]
            if got != value:
                raise RuntimeError(f"chain lesson {_op} {left} {right} wrote {got}")
            answers[index] = total
    keep = [index for index, answer in enumerate(answers) if answer is not None]
    prompts = []
    kept_answers = []
    kept_bins = []
    for index in keep:
        op, left, right, _value = rows[index]
        prompt = scratch._number(left) + [op] + scratch._number(right) + [scratch.EQ]
        if len(prompt) != PROMPT_LEN or answers[index] is None:
            raise RuntimeError("lesson prompt drifted")
        prompts.append(prompt)
        kept_answers.append(answers[index])
        kept_bins.append(bins[index])
    print(
        f"lessons {len(keep)}/{len(rows)} add {len(grouped[scratch.OP_ADD])} sub {len(grouped[scratch.OP_SUB])} div {len(grouped[scratch.OP_DIV])} larger {len(grouped[scratch.OP_LARGER])} smaller {len(grouped[scratch.OP_SMALLER])}",
        flush=True,
    )
    return (
        torch.tensor(prompts, dtype=torch.long, device=device),
        torch.tensor(kept_answers, dtype=torch.long, device=device),
        torch.tensor(kept_bins, dtype=torch.long, device=device),
    )


def _learn_loss(model: Place, prompt: torch.Tensor, answer: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    sign_logits, digit_logits, _hidden = model.core(*model._read_prompt(prompt))
    true_sign = (answer[:, 0] == MINUS_SIGN).long()
    true_digits = answer[:, 1:] - DIGIT0
    loss = F.nll_loss(digit_logits.reshape(-1, BASE), true_digits.reshape(-1))
    loss = loss + F.cross_entropy(sign_logits, true_sign)
    mask = bins >= 0
    if bool(mask.any()):
        loss = loss + F.cross_entropy(model._read_logits[mask], bins[mask])
    return loss


@torch.no_grad()
def _learned_drafts(model: Place, rows: list[tuple[int, int, int, int]], device: torch.device) -> list[int | None]:
    drafts = []
    for start in range(0, len(rows), scratch.BATCH):
        chunk = rows[start : start + scratch.BATCH]
        prompt = scratch._tensors(chunk, device)[:, :PROMPT_LEN]
        drafted, _sign, _digits, _hidden = model.predict_bundle(prompt)
        drafts.extend(scratch._read_number(draft) for draft in drafted.tolist())
    return drafts


def _place_digits(value: int) -> list[int]:
    digits = []
    remain = abs(value)
    for _ in range(PLACES):
        digits.append(remain % BASE)
        remain //= BASE
    return digits


@torch.no_grad()
def _sweep_trace(model: Place, op: int, left: int, right: int, device: torch.device) -> tuple[list[int], list[tuple[int, float]], list[int], int | None]:
    """Place totals the sweep named, the ones-place runner up, and the digits it wrote."""
    tokens, _sign, _digit_logits, _hidden = model.predict_bundle(_prompt(op, left, right, device))
    totals = [int(index) + VMIN for index in model._read_logits[0].argmax(dim=-1).tolist()]
    probs = torch.softmax(model._read_logits[0, 0], dim=-1)
    top = torch.topk(probs, k=2)
    ones = [(int(index) + VMIN, float(prob)) for index, prob in zip(top.indices.tolist(), top.values.tolist())]
    digits = (tokens[0, 1:] - DIGIT0).tolist()
    return totals, ones, digits, scratch._read_number(tokens[0].tolist())


def _explain_moves(
    model: Place,
    rows: list[tuple[int, int, int, int]],
    before: list[int | None],
    after: list[int | None],
    device: torch.device,
) -> None:
    """Group the far rows the lesson flipped and show the place totals behind a loss."""
    op = rows[0][0]
    lost = []
    gained = []
    for row, old, new in zip(rows, before, after):
        truth = row[3]
        if old == truth and new != truth:
            lost.append((row, new))
        elif old != truth and new == truth:
            gained.append((row, old))
    print(f"trace {NAME[op]} lost {len(lost)} gained {len(gained)}", flush=True)
    if op == scratch.OP_ADD:
        sum_ok = sum_hi = other = 0
        for (_op, left, right, truth), pred in lost:
            totals, _ones, _digits, _value = _sweep_trace(model, op, left, right, device)
            kinds = []
            for place in range(PLACES):
                pair = (left // BASE**place) % BASE + (right // BASE**place) % BASE
                if totals[place] == pair:
                    kinds.append("sum")
                elif totals[place] == pair + BASE:
                    kinds.append("plus10")
                else:
                    kinds.append("other")
            if "other" in kinds:
                other += 1
            elif "plus10" in kinds:
                sum_hi += 1
            else:
                sum_ok += 1
        print(f"trace + lost bins match {sum_ok} ten-high {sum_hi} other {other}", flush=True)
    if op == scratch.OP_SUB:
        borrowed = digit_ok = other = 0
        for (_op, left, right, truth), pred in lost:
            totals, _ones, digits, _value = _sweep_trace(model, op, left, right, device)
            raw = [_place_digits(left)[place] - _place_digits(right)[place] for place in range(PLACES)]
            if pred is not None and abs(pred) >= 1000:
                borrowed += 1
            elif digits == _place_digits(truth):
                digit_ok += 1
            elif totals == raw:
                other += 1
            else:
                other += 1
        print(f"trace - lost big {borrowed} digits-match {digit_ok} other {other}", flush=True)
    if op in (scratch.OP_LARGER, scratch.OP_SMALLER):
        chose_other = swapped = 0
        added = 0
        for (_op, left, right, truth), pred in lost:
            totals, _ones, _digits, _value = _sweep_trace(model, op, left, right, device)
            if pred not in (left, right):
                chose_other += 1
            else:
                swapped += 1
            pair = (left % BASE) + (right % BASE)
            diff = (left % BASE) - (right % BASE)
            if totals[0] in (pair, pair + BASE, diff):
                added += 1
        print(f"trace {NAME[op]} lost neither {chose_other} swapped {swapped} ones-was-arithmetic {added}", flush=True)
    shown = 0
    for (op_id, left, right, truth), pred in lost:
        if shown >= 6 and op != scratch.OP_DIV:
            break
        totals, ones, digits, value = _sweep_trace(model, op_id, left, right, device)
        print(
            f"trace {NAME[op]} {left} {right} truth {truth} pred {value} totals {totals} digits {digits} ones {ones[0][0]}:{ones[0][1]:.2f}/{ones[1][0]}:{ones[1][1]:.2f}",
            flush=True,
        )
        shown += 1


@torch.no_grad()
def _ones_total(model: Place, op: int, left: int, right: int, device: torch.device) -> int:
    model.core(*model._read_prompt(_prompt(op, left, right, device)))
    return int(model._read_logits[0, 0].argmax()) + VMIN


def _report_learn() -> None:
    """Teach the sweep the offset's place total, then score the far set once."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    device = torch.device("cuda")
    model = Place().to(device)
    path = DEST / "factor.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    model.reset_fast()
    _check_offset_bin(model, device)
    product = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.startswith(("prod_", "product", "mul_sign"))
    }
    banks = scratch._banks()
    before, before_n = _exact(model, banks["far"], device)
    print(f"baseline far {before}/{before_n} ops {_by_op(model, banks['far'], device)}", flush=True)
    traced_ops = (scratch.OP_ADD, scratch.OP_SUB, scratch.OP_DIV, scratch.OP_LARGER, scratch.OP_SMALLER)
    before_drafts = {
        op: _learned_drafts(model, [row for row in banks["far"] if row[0] == op], device) for op in traced_ops
    }
    for op, left, right in (
        (scratch.OP_ADD, 4, 5),
        (scratch.OP_ADD, 9, 9),
        (scratch.OP_SUB, 3, 7),
        (scratch.OP_ADD, 35, 37),
    ):
        _draft, pred, _called, _yes, _digits = _one(model, op, left, right, device)
        print(f"before {NAME[op]} {left} {right} pred {pred} ones { _ones_total(model, op, left, right, device)}", flush=True)
    prompts, answers, bins = _chain_lessons(model, banks["train"], device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(scratch.SEED)
    optimizer = torch.optim.AdamW(_sweep_parameters(model), lr=scratch.LR, weight_decay=0.01)
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(prompts), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(prompts), scratch.BATCH):
            pick = order[start : start + scratch.BATCH]
            optimizer.zero_grad(set_to_none=True)
            loss = _learn_loss(model, prompts[pick], answers[pick], bins[pick])
            if not torch.isfinite(loss):
                raise RuntimeError(f"learn loss is not finite at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_sweep_parameters(model), 1.0)
            optimizer.step()
            model.reset_fast()
            total += float(loss.detach()) * len(pick)
            seen += len(pick)
        print(f"learn epoch {epoch} loss {total / seen:.4f}", flush=True)
    model.eval()
    model.reset_fast()
    for name, param in model.named_parameters():
        if name in product and not torch.equal(param, product[name]):
            raise RuntimeError(f"learn moved {name}")
    after, after_n = _exact(model, banks["far"], device)
    print(f"learn far {after}/{after_n} ops {_by_op(model, banks['far'], device)}", flush=True)
    for op in traced_ops:
        rows = [row for row in banks["far"] if row[0] == op]
        after_drafts = _learned_drafts(model, rows, device)
        gained = lost = 0
        for row, old, new in zip(rows, before_drafts[op], after_drafts):
            truth = row[3]
            if old != truth and new == truth:
                gained += 1
            elif old == truth and new != truth:
                lost += 1
        print(f"learn move {NAME[op]} gained {gained} lost {lost}", flush=True)
        _explain_moves(model, rows, before_drafts[op], after_drafts, device)
    for op, left, right in (
        (scratch.OP_ADD, 4, 5),
        (scratch.OP_ADD, 9, 9),
        (scratch.OP_SUB, 3, 7),
        (scratch.OP_ADD, 35, 37),
        (scratch.OP_SUB, 31, 45),
        (scratch.OP_MUL, 37, 37),
        (scratch.OP_DIV, 40, 40),
    ):
        _draft, pred, _called, _yes, _digits = _one(model, op, left, right, device)
        print(f"after {NAME[op]} {left} {right} pred {pred} ones {_ones_total(model, op, left, right, device)}", flush=True)
    if after > before:
        torch.save(model.state_dict(), DEST / "dial.pt")
        print(f"wrote {DEST / 'dial.pt'}", flush=True)
    else:
        print("learn did not raise the far count", flush=True)
    print(f"read {path}", flush=True)


def _report_study() -> None:
    """Load the factor file, rehearse, and score. Does not write a checkpoint."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    device = torch.device("cuda")
    model = Place().to(device)
    path = DEST / "factor.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    model.reset_fast()
    keys = set(model.state_dict())
    params = sum(param.numel() for param in model.parameters())
    banks = scratch._banks()
    mul_rows = [row for row in banks["far"] if row[0] == scratch.OP_MUL]
    before, before_n = _exact(model, mul_rows, device)
    print(f"baseline far mul {before}/{before_n} params {params}", flush=True)
    _study(model, device)
    if set(model.state_dict()) != keys or "study_note" in model.state_dict():
        raise RuntimeError("study changed the saved tensors")
    hits, count = _exact(model, mul_rows, device)
    far_hits, far_n = _exact(model, banks["far"], device)
    print(f"study far mul {hits}/{count}", flush=True)
    print(f"study far {far_hits}/{far_n} ops {_by_op(model, banks['far'], device)}", flush=True)
    _probes(model, device)
    extra = ((7, 4), (37, 40), (38, 40), (37, 37), (39, 39), (36, 39))
    for left, right in extra:
        drafted, pred, _called, _yes, _digits = _one(model, scratch.OP_MUL, left, right, device)
        truth = scratch._truth(scratch.OP_MUL, left, right)
        print(f"probe {left}*{right} truth {truth} pred {pred} tokens {drafted}", flush=True)
    missed = 0
    for op, left, right, value in mul_rows:
        _drafted, pred, _called, _yes, _digits = _one(model, op, left, right, device)
        if pred != value:
            missed += 1
            print(f"miss {left}*{right} truth {value} pred {pred}", flush=True)
    print(f"study misses {missed}/{len(mul_rows)}", flush=True)
    print(f"read {path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "study":
        _report_study()
    elif len(sys.argv) > 1 and sys.argv[1] == "think":
        _report_think()
    elif len(sys.argv) > 1 and sys.argv[1] == "learn":
        _report_learn()
    else:
        main()
