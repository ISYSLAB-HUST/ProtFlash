""" 
"""
import torch
from .model import FLASHTransformer
from .utils import load_hub_workaround

MODEL_URL_BASE = "https://zenodo.org/record/7655858/files/protflash_large.pt"
MODEL_URL_SMALL = "https://zenodo.org/record/7655858/files/flash_protein.pt"


def load_prot_flash_base():
    model_data = load_hub_workaround(MODEL_URL_BASE)
    # model_data = torch.load("/mnt/d/protein-net/ProtBert/flash_protein.pt", map_location="cpu")
    hyper_parameter = model_data["hyper_parameters"]
    model = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layer'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])

    model.load_state_dict(model_data['state_dict'])

    return model

def load_prot_flash_small():
    model_data = load_hub_workaround(MODEL_URL_SMALL)
    # model_data = torch.load("/mnt/d/protein-net/ProtBert/flash_protein.pt", map_location="cpu")
    hyper_parameter = model_data["hyper_parameters"]
    model = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])

    model.load_state_dict(model_data['state_dict'])

    return model

if __name__ == "__main__":
    model = load_prot_flash_base()
