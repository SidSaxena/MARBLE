rm -rf output/probe.GTZANBeatTracking.MuQ.100hz
python cli.py fit -c configs/probe.MuQ.GTZANBeatTracking.100hz.yaml
python cli.py test -c configs/probe.MuQ.GTZANBeatTracking.100hz.yaml