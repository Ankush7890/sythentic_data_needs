#!/usr/bin/env bash
# Top up the toolace part sets after the `require` scaffold check was added: keep every row
# that passes it (--resume-from) and generate only the rest, same generator and prompts.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
for part in toolace_ops toolace_lookup toolace_roledef; do
    f="data/toolace_parts/highstakes_deepseekv4pro_tgtnone_${part}_600.jsonl"
    (
        $PY scripts/generate_split_targeted.py --concept highstakes --split "$part" \
            --model deepseek/deepseek-v4-pro --n 600 --no-shots \
            --resume-from "$f" --out "$f" > "logs/hardsplit_topup_${part}.log" 2>&1
        echo ">>> $(date -Is) $f: $(wc -l < "$f") rows"
    ) &
done
wait
echo ">>> $(date -Is) top-up finished."
