rm -rf output/probe.MTGInstrument.MusicFM
python cli.py fit -c configs/probe.MusicFM.MTGInstrument.yaml
python cli.py test -c configs/probe.MusicFM.MTGInstrument.yaml