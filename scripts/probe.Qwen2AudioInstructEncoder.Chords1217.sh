rm -rf output/probe.Chords1217.Qwen2AudioInstructEncoder
python cli.py fit -c configs/probe.Qwen2AudioInstructEncoder.Chords1217.yaml
python cli.py test -c configs/probe.Qwen2AudioInstructEncoder.Chords1217.yaml