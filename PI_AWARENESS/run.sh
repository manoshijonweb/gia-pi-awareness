#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing Python environment. Run bash install.sh first.' >&2; exit 1
fi
exec .venv/bin/python server.py "$@"
