""" 
"""
from typing import Sequence, Tuple, List, Union
from torch.nn.utils.rnn import pad_sequence
import torch
import pathlib
import urllib


residue_tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K',
                  'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', '<UNK>', 'PADDING_MASK', 'TOKEN_MASK']


token_to_index = {}
for i, j in enumerate(residue_tokens):
    token_to_index[j] = i


def batchConverter(raw_batch: Sequence[Tuple[str, str]]):
    ids = [item[0] for item in raw_batch]
    seqs = [item[1] for item in raw_batch]
    lengths = torch.tensor([len(item[1]) for item in raw_batch])
    batch_token = []
    for seq in seqs:
        batch_token.append(torch.tensor([token_to_index.get(i, token_to_index["<UNK>"]) for i in seq]))
    batch_token = pad_sequence(batch_token, batch_first=True, padding_value=token_to_index['PADDING_MASK'])
    return ids, batch_token, lengths


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = pathlib.Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check your network!")
    return data