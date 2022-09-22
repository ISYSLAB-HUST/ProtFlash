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
model(batch_token, lengths).shape
```

### License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.