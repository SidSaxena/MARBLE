rm -rf output/probe.MTGTop50.CLaMP3
torchrun --nproc_per_node=4  cli.py fit -c configs/probe.CLaMP3.MTGTop50.yaml
torchrun --nproc_per_node=4  cli.py test -c configs/probe.CLaMP3.MTGTop50.yaml