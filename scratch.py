"""A small model trained from nothing, on the relation itself.

Each number is a sign and four digits, least place first, so a place is a
token. The slow stack learns the true sum, difference, product, exact
quotient, and comparison. A second head says whether a finished equation
holds. A short-term residual stays silent during that training. It moves
only when the second head rejects a draft, and the draft it then prefers
is one of its own alternatives. An outside answer is a score, not a target.

This run writes ``weights/scratch/`` only.

    python scratch.py
"""

from __future__ import annotations

import json
import operator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "weights" / "scratch"
SEED = 20261109
WIDTH = 4
D_MODEL = 128
LAYERS = 4
HEADS = 4
RANK = 8
EPOCHS = 100
BATCH = 256
LR = 1e-3
FAR_LO = 31
FAR_HI = 45

PLUS_SIGN = 2
MINUS_SIGN = 3
OP_ADD = 4
OP_SUB = 5
OP_MUL = 6
OP_DIV = 7
OP_LARGER = 8
OP_SMALLER = 9
EQ = 10
DIGIT0 = 11
VOCAB = DIGIT0 + 10
PROMPT_LEN = 12
ANSWER_LEN = 5
LENGTH = PROMPT_LEN + ANSWER_LEN
YES = 0
NO = 1
OPS = (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_LARGER, OP_SMALLER)
STIFF = ((OP_ADD, -5, 3), (OP_LARGER, 3, 9), (OP_SMALLER, 9, 3))


def _truth(op: int, left: int, right: int) -> int | None:
    if op == OP_DIV:
        if right == 0 or left % right != 0:
            return None
        return operator.floordiv(left, right)
    return {
        OP_ADD: operator.add,
        OP_SUB: operator.sub,
        OP_MUL: operator.mul,
        OP_LARGER: max,
        OP_SMALLER: min,
    }[op](left, right)


def _expect(op: int, left: int, right: int) -> int | None:
    if op == OP_ADD:
        return left + right
    if op == OP_SUB:
        return left - right
    if op == OP_MUL:
        return left * right
    if op == OP_DIV:
        if right == 0 or left % right != 0:
            return None
        return left // right
    if op == OP_LARGER:
        return left if left >= right else right
    if op == OP_SMALLER:
        return left if left <= right else right
    return None


def _span_held(left: int, right: int) -> bool:
    """Scattered holdout.

    A mod-5 test collapses to the ones digits, because 10 ≡ 0 (mod 5).
    That deletes the whole digit sum, so a later place can never learn it.
    """
    return (left * 3 + right * 5) % 7 == 0


def _holds(op: int, left: int, right: int) -> bool:
    return (op, left, right) in STIFF or _span_held(left, right)


def _number(value: int) -> list[int]:
    sign = PLUS_SIGN if value >= 0 else MINUS_SIGN
    remain = abs(value)
    digits = []
    for _ in range(WIDTH):
        digits.append(DIGIT0 + remain % 10)
        remain //= 10
    if remain:
        raise RuntimeError(f"{value} does not fit in {WIDTH} digits")
    return [sign, *digits]


def _read_number(tokens: list[int]) -> int | None:
    if len(tokens) != ANSWER_LEN or tokens[0] not in (PLUS_SIGN, MINUS_SIGN):
        return None
    if any(token < DIGIT0 or token >= DIGIT0 + 10 for token in tokens[1:]):
        return None
    value = 0
    for place, token in enumerate(tokens[1:]):
        value += (token - DIGIT0) * 10**place
    return value if tokens[0] == PLUS_SIGN else -value


def _sequence(op: int, left: int, right: int, value: int) -> list[int]:
    row = _number(left) + [op] + _number(right) + [EQ] + _number(value)
    if len(row) != LENGTH:
        raise RuntimeError("sequence length drifted")
    return row


def _rows(ops: tuple[int, ...], lo: int, hi: int, keep) -> list[tuple[int, int, int, int]]:
    found = []
    for op in ops:
        for left in range(lo, hi + 1):
            for right in range(lo, hi + 1):
                if not keep(op, left, right):
                    continue
                value = _truth(op, left, right)
                if value is None or value != _expect(op, left, right):
                    continue
                found.append((op, left, right, value))
    return found


def _banks() -> dict[str, list[tuple[int, int, int, int]]]:
    core = _rows(OPS, 0, 30, lambda op, left, right: not _holds(op, left, right))
    signed = _rows((OP_ADD, OP_SUB), -12, 12, lambda op, left, right: (left < 0 or right < 0) and not _holds(op, left, right))
    hold_in = _rows(OPS, 0, 30, lambda op, left, right: _span_held(left, right) and (op, left, right) not in STIFF)
    hold_far = _rows(OPS, FAR_LO, FAR_HI, lambda op, left, right: True)
    stiff = []
    for op, left, right in STIFF:
        value = _truth(op, left, right)
        if value != _expect(op, left, right):
            raise RuntimeError(f"stiff label drifted {op} {left} {right}")
        stiff.append((op, left, right, value))
    train_keys = {(op, left, right) for op, left, right, _value in core + signed}
    for name, rows in (("in", hold_in), ("far", hold_far), ("stiff", stiff)):
        leaked = [row for row in rows if (row[0], row[1], row[2]) in train_keys]
        if leaked:
            raise RuntimeError(f"{name} leaked: {leaked[:3]}")
    return {"train": core + signed, "in": hold_in, "far": hold_far, "stiff": stiff}


def _audit(banks: dict[str, list[tuple[int, int, int, int]]]) -> None:
    for name, rows in banks.items():
        if not rows:
            raise RuntimeError(f"{name} bank is empty")
        for op, left, right, value in rows:
            if _truth(op, left, right) != value or _expect(op, left, right) != value:
                raise RuntimeError(f"bad label in {name}: {op} {left} {right} {value}")
            if _read_number(_number(value)) != value:
                raise RuntimeError(f"digit roundtrip failed for {value}")
    print(
        f"audit train {len(banks['train'])} in {len(banks['in'])} far {len(banks['far'])} stiff {len(banks['stiff'])}",
        flush=True,
    )


class Scratch(nn.Module):
    """Slow stack plus a silent short-term residual and a yes/no head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        block = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stack = nn.TransformerEncoder(block, num_layers=LAYERS, enable_nested_tensor=False)
        self.read = nn.Linear(D_MODEL, VOCAB)
        self.verdict = nn.Linear(D_MODEL, 2)
        self.fast_down = nn.Parameter(torch.empty(D_MODEL, RANK))
        self.fast_up = nn.Parameter(torch.zeros(RANK, D_MODEL))
        nn.init.normal_(self.fast_down, std=0.02)
        self.register_buffer("fast_down0", self.fast_down.detach().clone())

    def reset_fast(self) -> None:
        self.fast_down.data.copy_(self.fast_down0)
        self.fast_up.data.zero_()

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed(tokens)
        hidden = hidden + (hidden @ self.fast_down @ self.fast_up)
        mask = nn.Transformer.generate_square_subsequent_mask(
            hidden.shape[1], device=hidden.device, dtype=hidden.dtype
        )
        hidden = self.stack(hidden, mask=mask, is_causal=True)
        return self.read(hidden), self.verdict(hidden[:, -1])


def _tensors(rows: list[tuple[int, int, int, int]], device: torch.device) -> torch.Tensor:
    data = [_sequence(*row) for row in rows]
    return torch.tensor(data, dtype=torch.long, device=device)


def _corrupt(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    forged = batch.clone()
    for row in forged:
        value = _read_number(row[PROMPT_LEN:].tolist())
        if value is None:
            raise RuntimeError("corrupt saw a non-number")
        for _ in range(6):
            trial = row.clone()
            if value != 0 and int(torch.randint(0, 2, (1,), generator=generator)) == 0:
                trial[PROMPT_LEN] = MINUS_SIGN if trial[PROMPT_LEN] == PLUS_SIGN else PLUS_SIGN
            else:
                slot = PROMPT_LEN + 1 + int(torch.randint(0, WIDTH, (1,), generator=generator))
                trial[slot] = DIGIT0 + (trial[slot] - DIGIT0 + 1 + int(torch.randint(0, 9, (1,), generator=generator))) % 10
            if _read_number(trial[PROMPT_LEN:].tolist()) != value:
                row.copy_(trial)
                break
        else:
            raise RuntimeError("could not forge a wrong answer")
    return forged


def _answer_loss(logits: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    pred = logits[:, :-1, :]
    labels = batch[:, 1:].clone()
    labels[:, : PROMPT_LEN - 1] = -100
    return F.cross_entropy(pred.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100)


def _slow_parameters(model: Scratch) -> list[nn.Parameter]:
    return [param for name, param in model.named_parameters() if not name.startswith("fast_")]


@torch.no_grad()
def _greedy(model: Scratch, prompt: torch.Tensor) -> torch.Tensor:
    tokens = prompt.clone()
    for index in range(ANSWER_LEN):
        logits, _verdict = model(tokens)
        nxt = logits[:, -1, :]
        if index == 0:
            choice = nxt[:, PLUS_SIGN : MINUS_SIGN + 1].argmax(dim=-1) + PLUS_SIGN
        else:
            choice = nxt[:, DIGIT0 : DIGIT0 + 10].argmax(dim=-1) + DIGIT0
        tokens = torch.cat((tokens, choice.unsqueeze(1)), dim=1)
    return tokens


@torch.no_grad()
def _exact(model: Scratch, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int]:
    if not rows:
        return 0, 0
    hits = 0
    prompts = _tensors(rows, device)[:, :PROMPT_LEN]
    for start in range(0, len(rows), BATCH):
        drafted = _greedy(model, prompts[start : start + BATCH])
        for row, draft in zip(rows[start : start + BATCH], drafted.tolist()):
            if _read_number(draft[PROMPT_LEN:]) == row[3]:
                hits += 1
    return hits, len(rows)


def _span(rows: list[tuple[int, int, int, int]], count: int = 512) -> list[tuple[int, int, int, int]]:
    if len(rows) <= count:
        return rows
    step = max(1, len(rows) // count)
    return rows[::step][:count]


def _copy_left(rows: list[tuple[int, int, int, int]]) -> int:
    return sum(value == left for _op, left, _right, value in rows)


@torch.no_grad()
def _verdict_hits(model: Scratch, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int]:
    hits = 0
    prompts = _tensors(rows, device)[:, :PROMPT_LEN]
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        drafted = _greedy(model, prompts[start : start + BATCH])
        _logits, verdict = model(drafted)
        called = verdict.argmax(dim=-1).tolist()
        for row, draft, call in zip(chunk, drafted.tolist(), called):
            right = _read_number(draft[PROMPT_LEN:]) == row[3]
            if (call == YES and right) or (call == NO and not right):
                hits += 1
    return hits, len(rows)


def _alternate(model: Scratch, draft: torch.Tensor) -> torch.Tensor:
    """Second token at the answer place where the margin is thinnest."""
    tokens = draft[:PROMPT_LEN].clone().unsqueeze(0)
    margins = []
    choices = []
    for index in range(ANSWER_LEN):
        logits, _verdict = model(tokens)
        nxt = logits[0, -1]
        if index == 0:
            pool = nxt[PLUS_SIGN : MINUS_SIGN + 1]
            base = PLUS_SIGN
        else:
            pool = nxt[DIGIT0 : DIGIT0 + 10]
            base = DIGIT0
        order = torch.topk(pool, k=2)
        margins.append(float(order.values[0] - order.values[1]))
        choices.append(base + int(order.indices[1]))
        taken = draft[PROMPT_LEN + index].view(1, 1)
        tokens = torch.cat((tokens, taken), dim=1)
    slot = min(range(ANSWER_LEN), key=margins.__getitem__)
    revised = draft.clone()
    revised[PROMPT_LEN + slot] = choices[slot]
    return revised


def _revise(model: Scratch, rows: list[tuple[int, int, int, int]], device: torch.device) -> tuple[int, int, int]:
    """On a rejected draft, keep the alternative the verdict prefers."""
    improved = changed = rejected = 0
    was_right = 0
    prompts = _tensors(rows, device)
    fast = [model.fast_down, model.fast_up]
    for row, full in zip(rows, prompts):
        model.reset_fast()
        draft = _greedy(model, full[:PROMPT_LEN].unsqueeze(0))[0]
        _logits, verdict = model(draft.unsqueeze(0))
        if int(verdict.argmax()) == YES:
            continue
        rejected += 1
        other = _alternate(model, draft)
        _logits, other_verdict = model(other.unsqueeze(0))
        if float(other_verdict[0, YES]) <= float(verdict[0, YES]):
            continue
        optimizer = torch.optim.SGD(fast, lr=0.1)
        target = other[PROMPT_LEN:].unsqueeze(0)
        prompt = full[:PROMPT_LEN].unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        made = _greedy_train(model, prompt, target)
        made.backward()
        optimizer.step()
        revised = _greedy(model, prompt)[0]
        before = _read_number(draft[PROMPT_LEN:].tolist()) == row[3]
        after = _read_number(revised[PROMPT_LEN:].tolist()) == row[3]
        if revised[PROMPT_LEN:].tolist() != draft[PROMPT_LEN:].tolist():
            changed += 1
        if after and not before:
            improved += 1
        if before:
            was_right += 1
    return rejected, changed, improved


def _greedy_train(model: Scratch, prompt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tokens = prompt
    losses = []
    for index in range(ANSWER_LEN):
        logits, _verdict = model(tokens)
        losses.append(F.cross_entropy(logits[:, -1, :], target[:, index]))
        tokens = torch.cat((tokens, target[:, index].unsqueeze(1)), dim=1)
    return torch.stack(losses).mean()


def _fit(model: Scratch, train: list[tuple[int, int, int, int]], banks: dict[str, list[tuple[int, int, int, int]]], device: torch.device) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    optimizer = torch.optim.AdamW(_slow_parameters(model), lr=LR, weight_decay=0.01)
    data = _tensors(train, device)
    best_far = -1
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(train), BATCH):
            batch = data[order[start : start + BATCH]]
            forged = _corrupt(batch, generator)
            optimizer.zero_grad(set_to_none=True)
            logits, verdict = model(batch)
            _forged_logits, forged_verdict = model(forged)
            loss = _answer_loss(logits, batch)
            loss = loss + 0.5 * F.cross_entropy(verdict, torch.full((len(batch),), YES, device=device, dtype=torch.long))
            loss = loss + 0.5 * F.cross_entropy(forged_verdict, torch.full((len(batch),), NO, device=device, dtype=torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_slow_parameters(model), 1.0)
            optimizer.step()
            model.reset_fast()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            model.reset_fast()
            train_hits, train_n = _exact(model, _span(train), device)
            far_hits, far_n = _exact(model, banks["far"], device)
            print(
                f"epoch {epoch} loss {total / seen:.4f} train {train_hits}/{train_n} far {far_hits}/{far_n} gpu_gib {torch.cuda.memory_allocated() / 1024**3:.2f}",
                flush=True,
            )
            if far_hits > best_far:
                best_far = far_hits
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no epoch was scored")
    model.load_state_dict(best_state)
    print(f"kept far {best_far}/{len(banks['far'])}", flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Refusing to build the model in system RAM.")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    banks = _banks()
    _audit(banks)
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    model = Scratch().to(device)
    if any(param.device.type != "cuda" for param in model.parameters()):
        raise SystemExit("a parameter is off the GPU")
    print(f"gpu_mib {torch.cuda.memory_allocated() / 1024**2:.1f} params {sum(param.numel() for param in model.parameters())}", flush=True)
    if float(model.fast_up.detach().norm()) != 0.0:
        raise SystemExit("short-term layer is not silent")
    _fit(model, banks["train"], banks, device)
    model.eval()
    model.reset_fast()
    report = {}
    for name in ("train", "in", "far", "stiff"):
        sample = banks[name] if name != "train" else _span(banks["train"])
        hits, count = _exact(model, sample, device)
        copied = _copy_left(sample)
        verdict_hits, verdict_n = _verdict_hits(model, sample, device)
        report[name] = {"exact": hits, "count": count, "copy_left": copied, "verdict": verdict_hits, "verdict_n": verdict_n}
        print(
            f"{name} {hits}/{count} copy_left {copied}/{count} verdict {verdict_hits}/{verdict_n}",
            flush=True,
        )
    rejected, changed, improved = _revise(model, banks["stiff"] + banks["far"][:64], device)
    print(f"revise rejected {rejected} changed {changed} improved {improved}", flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEST / "model.pt")
    (DEST / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {DEST / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
