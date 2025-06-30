rm -rf output/probe.EMO.Xcodec
python cli.py fit -c configs/probe.Xcodec.EMO.yaml
python cli.py test -c configs/probe.Xcodec.EMO.yaml