#!/usr/bin/env python
"""Generate one 600-row set per prompt in a `## Prompt N` markdown file.

The prompts are written from measurements of ONE eval split (see
`TOOLACE_GENERATOR_PROMPTS.md`, `ANTHROPIC_HH_GENERATOR_PROMPTS.md`), and each asks for 20
rows per call, 10 per label, with the classes matched on surface features inside the call.
That is why a call carries both labels here, where `generate_split_targeted.py` asks for
one label per call.

The prompt text is read from the markdown file (the blockquote under `## Prompt N`), so
the file stays the single source of what was sent. Only one thing is appended per call:
the same "do not reuse these already-used openings" block `generate_split_targeted.py`
adds, with 20 openings sampled from rows this set has already kept.

Guards per row, as in `generate_split_targeted.py`:
  - shape: a leading system turn, then user/assistant alternation, ending on the
    assistant. Consecutive assistant turns (a tool trace, say) are MERGED with a blank
    line, which is what `combine_consecutive_messages` does before extraction anyway;
  - label: exactly `high-stakes` or `low-stakes`, at most n/2 per label;
  - length: <= 1024 gemma-3-27b tokens through `TokenBudget` (fails open);
  - novelty: the request's opening, with the ToolACE "Role definition" wrapper stripped
    so wrapper rows are not all keyed on the same boilerplate;
  - leakage: a row whose first user turn equals any dev or eval user turn of the split
    being written for is dropped.

Raw replies go to `logs/prompt_variants/<out stem>/`, one file per call.

    OPENROUTER_TIMEOUT_S=1800 .venv_claude/bin/python scripts/generate_prompt_variants.py \\
        --prompts-md ANTHROPIC_HH_GENERATOR_PROMPTS.md --split anthropic_hh_balanced \\
        --prompt 1 --out data/highstakes_deepseekv4pro_hh5_replica_600.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentic_redteam.json_extract import extract_json_values  # noqa: E402
from agentic_redteam.openrouter_client import (  # noqa: E402
    extract_openrouter_error,
    make_sync_client,
)

PROBE_MODEL = "google/gemma-3-27b-it"
LABELS = ("high-stakes", "low-stakes")
AVOID_SAMPLE = 20


def split_files(split: str) -> list[Path]:
    """The dev and eval files of the split being written for — the leak check reads these."""
    return [REPO / f"dev_samples/highstakes/{split}.jsonl",
            REPO / f"eval_sets/highstakes/{split}.jsonl"]


def load_prompt(prompts_md: Path, k: int) -> str:
    """The blockquote under `## Prompt k`, with the `> ` markers removed."""
    text = prompts_md.read_text(encoding="utf-8")
    parts = re.split(r"^## Prompt (\d+)[^\n]*\n", text, flags=re.M)
    by_num = {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    if k not in by_num:
        raise SystemExit(f"no '## Prompt {k}' in {prompts_md}")
    lines = []
    for line in by_num[k].splitlines():
        if line.startswith(">"):
            lines.append(line[2:] if line.startswith("> ") else line[1:])
        elif line.strip() == "---":
            break
    return "\n".join(lines).strip()


def request_text(row_msgs: list[dict]) -> str:
    """The user's request with the ToolACE history wrapper stripped."""
    u = next((m["content"] for m in row_msgs if m["role"] == "user"), "")
    # The wrapper's first "Inquirer:" is the role boilerplate; the request follows the history.
    m = re.search(r"Historical dialog data is as follows:\s*Inquirer:\s*(.*)", u, flags=re.S)
    return (m.group(1) if m else u).strip()


def opening(row_msgs: list[dict], with_reply: bool = False) -> str:
    """The novelty key for a row.

    By default the first user turn, which is what keeps a run's scenarios distinct. A prompt
    that deliberately writes one conversation TWICE — the same user turns with a different
    final assistant turn, as the HH `chosen`/`rejected` pairs do — needs the reply in the key
    too, or the second version is dropped as a duplicate of the first.
    """
    key = " ".join(request_text(row_msgs).split())[:200].lower()
    if with_reply:
        key += " || " + " ".join(row_msgs[-1]["content"].split())[:200].lower()
    return key


def _accept(value):
    def ok(d):
        return isinstance(d, dict) and isinstance(d.get("messages"), list) and "label" in d
    if isinstance(value, dict):
        if ok(value):
            return [value]
        inner = value.get("samples") or value.get("rows") or value.get("items")
        return _accept(inner) if isinstance(inner, list) else None
    if isinstance(value, list):
        got = [v for v in value if ok(v)]
        return got or None
    return None


def clean(item: dict) -> tuple[str, list[dict]] | None:
    """(label, messages) for a well-shaped row, else None."""
    label = str(item.get("label", "")).strip().lower()
    if label not in LABELS:
        return None
    msgs: list[dict] = []
    for m in item.get("messages") or []:
        if not isinstance(m, dict):
            return None
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role not in ("system", "user", "assistant") or not content:
            return None
        if msgs and msgs[-1]["role"] == role and role == "assistant":
            msgs[-1]["content"] += "\n\n" + content
        else:
            msgs.append({"role": role, "content": content})
    if len(msgs) < 3 or msgs[0]["role"] != "system" or msgs[-1]["role"] != "assistant":
        return None
    body = [m["role"] for m in msgs[1:]]
    if body != ["user", "assistant"] * (len(body) // 2) or len(body) % 2:
        return None
    return label, msgs


def _call(client, model, prompt, temperature, max_tokens, tag, raw_dir, tries=4):
    delay = 10.0
    for attempt in range(1, tries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — one dead call must not kill the run
            print(f"  [warn] {tag} attempt {attempt}: {type(exc).__name__}: {str(exc)[:160]}",
                  file=sys.stderr, flush=True)
            if attempt == tries:
                return []
            time.sleep(delay)
            delay *= 2
            continue
        if not getattr(resp, "choices", None):
            print(f"  [warn] {tag}: {extract_openrouter_error(resp) or 'no choices'}",
                  file=sys.stderr, flush=True)
            if attempt == tries:
                return []
            time.sleep(delay)
            continue
        text = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
        usage = getattr(resp, "usage", None)
        (raw_dir / f"{tag}_a{attempt}.txt").write_text(
            f"finish={finish} usage={usage} seconds={time.time() - t0:.0f}\n\n{text}",
            encoding="utf-8")
        items = [i for g in extract_json_values(text, _accept) for i in g]
        if items:
            return items
        print(f"  [warn] {tag}: no items parsed (finish={finish}) from {text[:140]!r}",
              file=sys.stderr, flush=True)
        if attempt == tries:
            return []
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-md", type=Path, default=REPO / "TOOLACE_GENERATOR_PROMPTS.md",
                    help="markdown file holding the `## Prompt N` blockquotes")
    ap.add_argument("--split", default="toolace_balanced",
                    help="the eval split being written for; its dev and eval files are the "
                         "leak check")
    ap.add_argument("--prompt", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--max-sample-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dup-key", choices=["user", "user+reply"], default="user",
                    help="what makes a row a duplicate: its first user turn (default), or "
                         "that plus the final assistant turn — needed by a prompt that writes "
                         "each conversation twice with different replies")
    ap.add_argument("--dump-prompt", action="store_true", help="print the prompt and exit")
    args = ap.parse_args()

    base_prompt = load_prompt(args.prompts_md, args.prompt)
    if args.dump_prompt:
        print(base_prompt)
        return

    budget = None
    if args.max_sample_tokens > 0:
        from agentic_redteam.token_budget import TokenBudget
        budget = TokenBudget(PROBE_MODEL, args.max_sample_tokens,
                             combine_consecutive_messages=True, convert_tool_to_assistant=True)
        budget.warmup()

    split_users = set()
    for f in split_files(args.split):
        for line in f.open(encoding="utf-8"):
            msgs = json.loads(json.loads(line)["inputs"])
            split_users.add(next((m["content"].strip() for m in msgs if m["role"] == "user"), ""))

    raw_dir = REPO / "logs/prompt_variants" / args.out.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = make_sync_client()
    rng = random.Random(args.seed)
    per_label = args.n // 2
    rows: list[dict] = []
    seen: dict[str, None] = {}
    counts = {lab: 0 for lab in LABELS}
    dropped = {"shape": 0, "long": 0, "dup": 0, "leak": 0, "full": 0}
    lock = threading.Lock()
    calls = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while min(counts.values()) < per_label and calls < args.max_calls:
            futures = []
            for _ in range(min(args.concurrency, args.max_calls - calls)):
                used = list(seen)
                sample = rng.sample(used, AVOID_SAMPLE) if len(used) > AVOID_SAMPLE else used
                prompt = base_prompt
                if sample:
                    prompt += ("\n\nDo NOT reuse or lightly reword these already-used requests; "
                               "pick clearly different scenarios:\n"
                               + "\n".join(f"- {a}" for a in sample))
                tag = f"call{calls:03d}"
                futures.append(pool.submit(_call, client, args.model, prompt, args.temperature,
                                           args.max_tokens, tag, raw_dir))
                calls += 1
            for fut in futures:
                for item in fut.result():
                    got = clean(item)
                    if got is None:
                        dropped["shape"] += 1
                        continue
                    label, msgs = got
                    if budget is not None and budget.overage(msgs) is not None:
                        dropped["long"] += 1
                        continue
                    if next(m["content"] for m in msgs if m["role"] == "user") in split_users:
                        dropped["leak"] += 1
                        continue
                    key = opening(msgs, args.dup_key == "user+reply")
                    with lock:
                        if not key or key in seen:
                            dropped["dup"] += 1
                            continue
                        if counts[label] >= per_label:
                            dropped["full"] += 1
                            continue
                        seen[key] = None
                        counts[label] += 1
                        rows.append({"inputs": json.dumps(msgs, ensure_ascii=False),
                                     "labels": label})
            print(f"  p{args.prompt}: {len(rows)}/{args.n} "
                  f"({counts['high-stakes']}/{counts['low-stakes']}) after {calls} calls; "
                  f"dropped {dropped}", file=sys.stderr, flush=True)

    if min(counts.values()) < per_label:
        print(f"  [warn] stopped at the call budget with {counts}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}  (dropped {dropped})", file=sys.stderr)


if __name__ == "__main__":
    main()
