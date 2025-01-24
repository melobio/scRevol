### Run the code
For simulation data, make sure the dataset starts with a number, such as data_large2：\${id}_clone\${i}_error\${j}, which can be run quickly by number. For example, to run dataset 0 (0_clone6_error0.4) on the simulated data data_large2, use the following code:

```python
python train.py \
    --data_dir ../data/data_large2 \ # The path of simulation folder
    --dataset_id 0 \                 # Simulation data number, 0 is to access the dataset starting with number 0 in this folder
    --output_dir results_leiden \    # Output folder name, internally store the training process and results according to the simulation number
```
The remaining parameters can use the default values.

For real data, datasets for which real labels cannot be obtained, access the data by file name：
```python
python train.py \
    --data_dir ../data/lineage_trace_data/lineage_trace_data/ \ # The path of folder
    --data_name c17_CNV \                                       # Folder name
    --output_dir result_lin \                                   # The path of output
    --eval_mode 
```

Note that the simulation data is in tsv format, and the real data is in csv format. If the simulation data is converted to csv, you can also use the second method to access.

### View the experimental results
* Training process：View through tensorboard
* Results：After the training is completed, the following output can be found in the output folder：
    * cell2cluster.csv：The category number to which each cell belongs.
    * tree_path.csv： The tree generation path, where each row has a parent node on the left and a child node on the right, stored using a depth-first algorithm,
    * tree.png：Schematic diagram of a spanning tree.

