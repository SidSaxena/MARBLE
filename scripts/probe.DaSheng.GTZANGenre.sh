rm -rf output/probe.GTZANGenre.DaSheng
python cli.py fit -c configs/probe.DaSheng.GTZANGenre.yaml --optimizer.lr=1e-3
python cli.py test -c configs/probe.DaSheng.GTZANGenre.yaml