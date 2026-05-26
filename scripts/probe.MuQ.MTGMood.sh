#!/usr/bin/env bash
set -euo pipefail

rm -rf output/probe.MTGMood.MuQ
python cli.py fit -c configs/probe.MuQ.MTGMood.yaml
python cli.py test -c configs/probe.MuQ.MTGMood.yaml
