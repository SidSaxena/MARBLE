rm -rf output/probe.EMO.MuQ
python cli.py fit -c configs/probe.MuQ.EMO.yaml
python cli.py test -c configs/probe.MuQ.EMO.yaml