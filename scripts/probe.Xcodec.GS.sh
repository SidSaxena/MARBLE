rm -rf output/probe.GS.Xcodec
python cli.py fit -c configs/probe.Xcodec.GS.yaml
python cli.py test -c configs/probe.Xcodec.GS.yaml