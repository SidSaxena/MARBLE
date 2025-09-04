rm -rf output/probe.GS.ChromaTA
python cli.py fit -c configs/probe.Chroma.GS.yaml
python cli.py test -c configs/probe.Chroma.GS.yaml