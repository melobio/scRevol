source activate clone

task=CNV_multiSample_martix

datadir=../data/${task}

type=CNV
                                                                                            # task
# for i in c17 c22 c26 c29 c30 c39                                                            # lineage_trace_data
for i in data2filtered data5filtered data6filtered data7filtered                                              # CNV_multiSample_martix
# for i in GSM5276940 GSM5276943 SOL003 SOL006 SOL008 SOL012 SOL016 SOL1303 SOL1306 SOL1307 # SingleSample_CNV
do
    python train.py \
    --data_dir ${datadir} \
    --data_name ${i}_${type} \
    --output_dir results/${task}/rl_leiden/${type} \
    --eval_mode \
    --meta_col sample
done