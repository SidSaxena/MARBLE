rm -rf output/probe.MTGTop50.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.MTGTop50.yaml
python cli.py test -c configs/probe.MERT-v1-95M.MTGTop50.yaml