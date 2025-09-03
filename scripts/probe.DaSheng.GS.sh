rm -rf output/probe.GS.DaSheng
python cli.py fit -c configs/probe.DaSheng.GS.yaml
python cli.py test -c configs/probe.DaSheng.GS.yaml