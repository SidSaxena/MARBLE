rm -rf output/probe.HookTheoryKey.MERT-v1-95M
CUDA_VISIBLE_DEVICES=2 python cli.py fit -c configs/probe.MERT-v1-95M.HookTheoryKey.yaml
CUDA_VISIBLE_DEVICES=2 python cli.py test -c configs/probe.MERT-v1-95M.HookTheoryKey.yaml