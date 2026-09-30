#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt -c constraints.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m app.train --demo
.venv/bin/python -m app.batch
printf '%s\n' 'Read outputs/predictions.jsonl and artifacts/evaluation.json'
