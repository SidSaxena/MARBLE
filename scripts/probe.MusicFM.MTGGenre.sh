#!/usr/bin/env bash
set -euo pipefail

rm -rf output/probe.MTGGenre.MusicFM
python cli.py fit -c configs/probe.MusicFM.MTGGenre.yaml
python cli.py test -c configs/probe.MusicFM.MTGGenre.yaml
