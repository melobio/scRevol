source activate clone


for method in rl_leiden
do
    for data_type in CNV
    do
        # python eval_sample.py   --task_type CNV_multiSample_martix \
        #                         --method ${method} \
        #                         --data_type ${data_type} \
        #                         --data_names data1,data2,data5,data6,data7 \
        #                         --meta_columns celltype # ,Stage,sample,anatomical_location
        python eval_sample.py   --task_type SingleSample_CNV \
                                --method ${method} \
                                --data_type ${data_type} \
                                --data_names GSM5276940filtered,GSM5276943filtered,SOL1303filtered,SOL1306filtered,SOL003filtered,SOL006filtered,SOL008filtered,SOL012filtered,SOL016filtered \
                                --meta_columns celltype
    done
done