### 运行代码
对于仿真数据，确保数据集内部使用编号开头，比如data_large2的组织形式：\${id}_clone\${i}_error\${j}，可以通过编号快速运行。比如在仿真数据data_large2上运行0号数据集（0_clone6_error0.4）则使用以下代码：

```python
python train.py \
    --data_dir ../data/data_large2 \ # 仿真文件夹路径
    --dataset_id 0 \                 # 仿真数据编号，0为访问该文件夹中以编号0作为开头的数据集
    --output_dir results_leiden \    # 输出文件夹名，内部按照仿真编号存储训练过程与结果
```
其余参数可以使用默认值。

对于真实数据，即无法获取真实标签的数据集，通过文件名访问数据：
```python
python train.py \
    --data_dir ../data/lineage_trace_data/lineage_trace_data/ \ # 文件夹
    --data_name c17_CNV \                                       # 文件名
    --output_dir result_lin \                                   # 输出地址
    --eval_mode 
```

注意仿真数据为tsv格式，真实数据为csv格式。如果将仿真数据转化为csv，则也可以使用第二种方法访问。

### 查看实验结果
* 训练过程：通过tensorboard查看
* 实验结果：训练结束后，在输出文件夹中可以找到以下输出内容：
    * cell2cluster.csv：每个细胞所属类别编号
    * tree_path.csv： 树的生成路径，其中每行左边为父节点，右边为子节点，使用深度优先算法存储
    * tree.png：生成树示意图

