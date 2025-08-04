rm -rf output/probe.MTGGenre.CLaMP3
torchrun --nproc_per_node=4  cli.py fit -c configs/probe.CLaMP3.MTGGenre.yaml
torchrun --nproc_per_node=4  cli.py test -c configs/probe.CLaMP3.MTGGenre.yaml