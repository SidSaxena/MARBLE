rm -rf output/probe.MTGMood.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.MTGMood.yaml
python cli.py test -c configs/probe.MERT-v1-95M.MTGMood.yaml