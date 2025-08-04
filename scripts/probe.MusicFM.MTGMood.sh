rm -rf output/probe.MTGMood.MusicFM
python cli.py fit -c configs/probe.MusicFM.MTGMood.yaml
python cli.py test -c configs/probe.MusicFM.MTGMood.yaml