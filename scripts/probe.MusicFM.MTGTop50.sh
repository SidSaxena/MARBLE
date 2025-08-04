rm -rf output/probe.MTGTop50.MusicFM
python cli.py fit -c configs/probe.MusicFM.MTGTop50.yaml
python cli.py test -c configs/probe.MusicFM.MTGTop50.yaml