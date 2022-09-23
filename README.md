## ProtFlash: A lightweight protein language model

### Install 
As a prerequisite, you must have PyTorch installed to use this repository.

You can use this one-liner for installation, using the latest release version
```
pip install git+https://github.com/isyslab-hust/ProtFlash
```
### Usage
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

### License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.