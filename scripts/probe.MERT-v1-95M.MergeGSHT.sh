rm -rf output/probe.MergeGSHT.MERT-v1-95M
CUDA_VISIBLE_DEVICES=2 python cli.py fit -c configs/probe.MERT-v1-95M.MergeGSHT.yaml
CUDA_VISIBLE_DEVICES=2 python cli.py test -c configs/probe.MERT-v1-95M.MergeGSHT.yaml