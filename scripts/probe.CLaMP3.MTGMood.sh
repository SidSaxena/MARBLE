rm -rf output/probe.MTGMood.CLaMP3
torchrun --nproc_per_node=4  cli.py fit -c configs/probe.CLaMP3.MTGMood.yaml
torchrun --nproc_per_node=4  cli.py test -c configs/probe.CLaMP3.MTGMood.yaml