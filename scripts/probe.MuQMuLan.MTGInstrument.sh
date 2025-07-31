rm -rf output/probe.MTGInstrument.MuQMuLan
python cli.py fit -c configs/probe.MuQMuLan.MTGInstrument.yaml
python cli.py test -c configs/probe.MuQMuLan.MTGInstrument.yaml