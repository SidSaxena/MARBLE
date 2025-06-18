rm -rf output/probe.GTZANBeatTracking.MuQ
python cli.py fit -c configs/probe.MuQ.GTZANBeatTracking.yaml
python cli.py test -c configs/probe.MuQ.GTZANBeatTracking.yaml