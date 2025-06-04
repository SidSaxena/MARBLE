rm -rf output/probe.Qwen2AudioInstructEncoder.GTZANBeatTracking
python cli.py fit -c configs/probe.Qwen2AudioInstructEncoder.GTZANBeatTracking.yaml
python cli.py test -c configs/probe.Qwen2AudioInstructEncoder.GTZANBeatTracking.yaml