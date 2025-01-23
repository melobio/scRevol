source activate clone

for i in {413..499}
do
    python train.py --algo leiden --dataset_id $i --output_dir results/data_large2/rl_leiden/CNV
done