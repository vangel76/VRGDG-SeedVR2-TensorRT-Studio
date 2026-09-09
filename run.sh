#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo 'The Linux virtual environment is missing. Run scripts/install.sh first.' >&2
    exit 1
fi
echo 'Open http://127.0.0.1:7870/ in your browser. Press Ctrl+C to stop Studio.'
exec .venv/bin/python -m uvicorn api_server:app --host 127.0.0.1 --port 7870 "$@"
