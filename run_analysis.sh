#!/bin/bash
cd "$(dirname "$0")"
export FLASK_APP=app.py
export FLASK_ENV=production
python3 app.py
