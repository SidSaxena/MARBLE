rm -rf output/probe.GTZANBeatTracking.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.GTZANBeatTracking.yaml
python cli.py test -c configs/probe.MERT-v1-95M.GTZANBeatTracking.yaml