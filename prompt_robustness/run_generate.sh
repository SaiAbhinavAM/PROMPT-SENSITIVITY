#!/bin/zsh
cd "$(dirname "$0")"
PYTHONUNBUFFERED=1 ./venv/bin/python3.9 -u generate_dataset.py > /tmp/gensens_run.log 2>&1
echo "EXIT:$?" >> /tmp/gensens_run.log
