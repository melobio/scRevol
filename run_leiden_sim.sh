source activate clone

for i in {1..499}
do
    python train.py --algo leiden --dataset_id $i --output_dir results/data_large2/leiden/CNV --rl_epoch 0
done