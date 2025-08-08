rm -rf output/probe.MTT.MuQMuLan
torchrun --nproc_per_node=4 cli.py fit -c configs/probe.MuQMuLan.MTT.yaml
torchrun --nproc_per_node=4 cli.py test -c configs/probe.MuQMuLan.MTT.yaml