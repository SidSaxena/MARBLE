rm -rf output/probe.MTT.MusicFM
torchrun --nproc_per_node=4 cli.py fit -c configs/probe.MusicFM.MTT.yaml
torchrun --nproc_per_node=4 cli.py test -c configs/probe.MusicFM.MTT.yaml