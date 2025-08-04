rm -rf output/probe.MTGTop50.MuQMuLan
python cli.py fit -c configs/probe.MuQMuLan.MTGTop50.yaml
python cli.py test -c configs/probe.MuQMuLan.MTGTop50.yaml