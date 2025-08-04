lr_list=(1e-3 5e-3 5e-4 1e-2 1e-4 5e-5)
TASKS=(MTGInstrument MTGMood MTGTop50 MTGGenre)

for lr_value in ${lr_list[@]}; do
    for ((i=0; i<${#TASKS[@]}; i++)); do
        torchrun --master_port=12345 --nproc_per_node=4 cli.py fit -c configs/probe.CLaMP3.${TASKS[i]}.yaml --optimizer.lr=${lr_value}
        torchrun --master_port=12345 --nproc_per_node=4 cli.py test -c configs/probe.CLaMP3.${TASKS[i]}.yaml
    done
done