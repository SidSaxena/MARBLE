#!/usr/bin/env bash
set -euo pipefail

rm -rf output/probe.MergeGSHT.MuQ
python cli.py fit -c configs/probe.MuQ.MergeGSHT.yaml
python cli.py test -c configs/probe.MuQ.MergeGSHT.yaml
