rm -rf output/probe.GS.ChromaTA
python cli.py fit -c configs/probe.ChromaTA.GS.yaml
python cli.py test -c configs/probe.ChromaTA.GS.yaml