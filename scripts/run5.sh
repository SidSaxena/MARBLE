bash scripts/probe.MuQMuLan.MTT.sh
bash scripts/probe.CLaMP3.MTT.sh
bash scripts/probe.MusicFM.MTT.sh
bash scripts/probe.MuQ.MTT.sh

lr_list=(5e-3 5e-4 1e-2 1e-4 5e-5 1e-3)

for lr_value in ${lr_list[@]}; do
    torchrun --master_port=12346 --nproc_per_node=4 cli.py fit -c configs/probe.MuQMuLan.MTT.yaml --optimizer.lr=${lr_value}
    torchrun --master_port=12346 --nproc_per_node=4 cli.py test -c configs/probe.MuQMuLan.MTT.yaml
done