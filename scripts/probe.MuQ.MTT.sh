rm -rf output/probe.MTT.MuQ
torchrun --nproc_per_node=4 cli.py fit -c configs/probe.MuQ.MTT.yaml
torchrun --nproc_per_node=4 cli.py test -c configs/probe.MuQ.MTT.yaml