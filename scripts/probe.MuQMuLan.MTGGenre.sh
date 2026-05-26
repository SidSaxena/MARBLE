#!/usr/bin/env bash
set -euo pipefail

rm -rf output/probe.MTGGenre.MuQMuLan
python cli.py fit -c configs/probe.MuQMuLan.MTGGenre.yaml
python cli.py test -c configs/probe.MuQMuLan.MTGGenre.yaml
