rm -rf output/probe.Chords1217.MERT-v1-95M
python cli.py fit -c configs/probe.MERT-v1-95M.Chords1217.yaml
python cli.py test -c configs/probe.MERT-v1-95M.Chords1217.yaml