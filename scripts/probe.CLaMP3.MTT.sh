rm -rf output/probe.MTT.CLaMP3
torchrun --nproc_per_node=4 cli.py fit -c configs/probe.CLaMP3.MTT.yaml
torchrun --nproc_per_node=4 cli.py test -c configs/probe.CLaMP3.MTT.yaml