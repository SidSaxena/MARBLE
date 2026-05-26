#!/usr/bin/env bash
set -euo pipefail

rm -rf output/probe.MTGTop50.MuQ
python cli.py fit -c configs/probe.MuQ.MTGTop50.yaml
python cli.py test -c configs/probe.MuQ.MTGTop50.yaml
