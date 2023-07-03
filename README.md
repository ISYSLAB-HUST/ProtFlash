## ProtFlash: A lightweight protein language model
[![PyPI - Version](https://img.shields.io/pypi/v/ProtFlash.svg?style=flat)](https://pypi.org/project/ProtFlash/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ProtFlash.svg)](https://pypi.org/project/ProtFlash/) [![GitHub - LICENSE](https://img.shields.io/github/license/isyslab-hust/ProtFlash.svg?style=flat)](./LICENSE) ![PyPI - Downloads](https://img.shields.io/pypi/dm/ProtFlash) [![Wheel](https://img.shields.io/pypi/wheel/ProtFlash)](https://pypi.org/project/ProtFlash/) ![build](https://img.shields.io/github/actions/workflow/status/isyslab-hust/ProtFlash/publish_to_pypi.yml)

### Install 
As a prerequisite, you must have PyTorch installed to use this repository.

You can use this one-liner for installation, using the latest release version
```
# latest version
pip install git+https://github.com/isyslab-hust/ProtFlash

# stable version
pip install ProtFlash
```

## **Model details**
|   **Model**    | **# of parameters** | **# of hidden size** |            **Pretraining dataset**             | **# of proteins** | **Model download** |
|:--------------:|:-------------------:|:----------------------:|:----------------------------------------------:|:-----------------:|:------------------------:|
|    ProtFlash-base    |        174M         |           768           | [UniRef100](https://www.uniprot.org/downloads) |       51M        |      [ProtFlash-base](https://zenodo.org/record/7655858/files/protflash_large.pt)       |
| ProtFlash-small |        79M         |           512           | [UniRef50](https://www.uniprot.org/downloads)  |        51M        |     [ProtFlash-small](https://zenodo.org/record/7655858/files/flash_protein.pt)     |

### Usage

#### protein sequence embedding
```
from ProtFlash.pretrain import load_prot_flash_base
from ProtFlash.utils import batchConverter
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
]
ids, batch_token, lengths = batchConverter(data)
model = load_prot_flash_base()
with torch.no_grad():
    token_embedding = model(batch_token, lengths)
# Generate per-sequence representations via averaging
sequence_representations = []
for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_embedding[i, 0: len(seq) + 1].mean(0))
```

#### loading weight files
```
import torch
from ProtFlash.model import FLASHTransformer

model_data = torch.load(your_parameters_file)
hyper_parameter = model_data["hyper_parameters"]
model = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter         ['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])

model.load_state_dict(model_data['state_dict'])
```

### License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

### Citation
If you use this code or one of our pretrained models for your publication, please cite our paper:
```
Lei Wang, Hui Zhang, Wei Xu, Zhidong Xue, and Yan Wang. ProtFlash: Deciphering the protein landscape with a novel and lightweight language model, Under revision (2023)
```