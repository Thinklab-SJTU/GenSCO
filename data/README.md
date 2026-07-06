# Data

All datasets can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1eyRkLda-y6VLCgpw0Tfsyw?pwd=fd59). Below we provide instructions for converting the data into the required format.

**TSP**

Download the txt files from https://github.com/Thinklab-SJTU/Fast-T2T/tree/main/data, and run:
```
names=(
    "tsp100_concorde_7.75585"
    "tsp500_concorde_16.54581"
    "tsp1000_concorde_23.11812"
    "tsp100_uniform_1.28m_lkh5k"
    "tsp500_uniform_train_lkh5w"
)
for name in "${names[@]}"; do
    python data/tsp_convert_fmt.py \
        --src-path datasets/${name}.txt \
        --tgt-path datasets/${name}.npz
done
```

**MIS**

For the original data, follow the instructions at https://github.com/Thinklab-SJTU/Fast-T2T/tree/main/data. Then execute:

```
python -u mis_convert.py \
    --src_path mis/rb/rb200_300_test \
    --save_path mis/rb_200_300_test.npz
python -u mis_convert.py \
    --src_path mis/rb/rb200_300_train \
    --save_path mis/rb_200_300_train.npz
python -u mis_convert.py \
    --src_path mis/er/er_700_800_test \
    --save_path mis/er_700_800_test.npz
python -u mis_convert.py \
    --src_path mis/er/er_700_800_train \
    --save_path mis/er_700_800_train.npz
```

In the MIS ER training set, sample `115672` should be deleted because its `found_mis` flag is `False`. Run the following Python code:
```
import json
import numpy as np

dataset = dict(np.load('mis/er_700_800_train.npz'))

ids = dataset['ids']
edges = dataset['edges']
num_nodes = dataset['num_nodes']

delete_id = 115672
keep = ids != delete_id
ids = ids[keep]
edges = edges[keep]
num_nodes = num_nodes[keep]

np.savez_compressed('mis/er_700_800_train_repaired.npz', ids=ids, num_nodes=num_nodes, edges=edges)
```

To convert the labels, run:
```
import json
import numpy as np

num_instances = 90000

labels_list = []
for i in range(num_instances):
    with open(f'mis/rb/rb200_300_train_label/GR_200_300_{i}_unweighted.result', 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x: bool(int(x.strip())), lines))
        labels = np.array(lines, dtype='bool')
        labels_list.append(labels)

num_nodes_padded = max(list(map(lambda x: x.size, labels_list)))

labels_list = list(map(lambda x: np.pad(x, pad_width=[(0, num_nodes_padded - x.size)], mode='constant'), labels_list))
labels_array = np.stack(labels_list, axis=0)

np.savez_compressed('mis/rb_200_300_train_labels.npz', labels=labels_array)
```
and
```
import json
import numpy as np

with open('mis/er/er_700_800_train_labels/results.json', 'r') as f:
    labels = json.load(f)

new_labels = {}
for k, v in labels.items():
    new_k = int(k.split('_')[-1])
    new_labels[new_k] = v
labels = new_labels

num_instances = len(labels)

labels_array = np.zeros([num_instances, 800], dtype='bool')

for k, v in labels.items():
    if v['found_mis']:
        labels_array[k][v['mis']] = True

np.savez_compressed('mis/er_700_800_train_repaired_labels.npz', labels=labels_array)
```

