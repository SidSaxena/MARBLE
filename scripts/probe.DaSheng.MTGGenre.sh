rm -rf output/probe.MTGGenre.DaSheng
torchrun --nproc_per_node=4  cli.py fit -c configs/probe.DaSheng.MTGGenre.yaml
torchrun --nproc_per_node=4  cli.py test -c configs/probe.DaSheng.MTGGenre.yaml