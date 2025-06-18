rm -rf output/probe.GS.MuQ
python cli.py fit -c configs/probe.MuQ.GS.yaml
python cli.py test -c configs/probe.MuQ.GS.yaml