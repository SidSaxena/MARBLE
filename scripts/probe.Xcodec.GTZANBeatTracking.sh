rm -rf output/probe.GTZANBeatTracking.Xcodec
python cli.py fit -c configs/probe.Xcodec.GTZANBeatTracking.yaml
python cli.py test -c configs/probe.Xcodec.GTZANBeatTracking.yaml