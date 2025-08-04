rm -rf output/probe.MTGInstrument.MuQ
python cli.py fit -c configs/probe.MuQ.MTGInstrument.yaml
python cli.py test -c configs/probe.MuQ.MTGInstrument.yaml