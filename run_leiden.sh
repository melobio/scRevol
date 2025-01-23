source activate clone

task=SingleSample_CNV

datadir=../data/${task}

type=expr

# for i in c17 c22 c26 c29 c30 c39 # lineage_trace_data
# for i in data1 data2 data3 data5 data6 data7  # data1 # CNV_multiSample_martix
for i in GSM5276940 GSM5276943 SOL003 SOL006 SOL008 SOL012 SOL016 SOL1303 SOL1306 SOL1307 # SingleSample_CNV
do
    python train.py \
    --data_dir ${datadir} \
    --data_name ${i}_${type} \
    --output_dir results/${task}/leiden/${type}_nonepi \
    --eval_mode \
    --rl_epochs 0 \
    --remove_non_epi
done