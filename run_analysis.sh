#!/bin/bash
cd "$(dirname "$0")"
export FLASK_APP=main.py
export FLASK_ENV=production
python3 main.py
