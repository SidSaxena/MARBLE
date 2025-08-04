rm -rf output/probe.MTGGenre.MuQ
python cli.py fit -c configs/probe.MuQ.MTGGenre.yaml
python cli.py test -c configs/probe.MuQ.MTGGenre.yaml