source activate clone

method=rl_leiden
python eval_sample.py --task_type lineage_trace_data --method ${method} --data_type CNV --data_names concat --meta_columns sample
python eval_lineage.py --method ${method}
python eval_lineage.py --method ${method} --concat