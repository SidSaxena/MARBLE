rm -rf output/probe.MTGInstrument.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.MTGInstrument.yaml
python cli.py test -c configs/probe.MERT-v1-95M.MTGInstrument.yaml