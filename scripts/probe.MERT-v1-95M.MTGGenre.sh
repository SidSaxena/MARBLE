rm -rf output/probe.MTGGenre.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.MTGGenre.yaml
python cli.py test -c configs/probe.MERT-v1-95M.MTGGenre.yaml