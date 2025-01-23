type=CNV

for d in data1 data2 data5 data6 data7 # data3 要超过一天
do
    python train.py --data_dir ../data/CNV_multiSample_martix --data_name ${d}_${type} --output_dir results/CNV_multiSample_martix/rl_leiden/${type} --eval_mode --meta_col sample
done
wait