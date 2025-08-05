rm -rf output/probe.MTT.MERT-v1-95M
torchrun --nproc_per_node=4 cli.py fit -c configs/probe.MERT-v1-95M.MTT.yaml
torchrun --nproc_per_node=4 cli.py fit -c configs/probe.MERT-v1-95M.MTT.yaml