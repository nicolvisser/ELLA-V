dependencies = ["torch", "torchaudio", "xformers", "simple_parsing", "tqdm"]

import torch

from ellav.model import ELLAVGARModel
from ellav.tokenizer import ELLAVTokenizer
from ellav.transformer.config import TransformerModelArgs

release_url = (
    "https://github.com/nicolvisser/ELLA-V/releases/download/v0.1.0/"
)

# fmt: off

ellav_phone_to_wavtokenizer_url = release_url + "ella-v-phonemes-us-arpa-clean-10k-40hz-e06af002.pt"


ellav_unit_to_wavtokenizer_urls = {
    (0,0):       release_url + "ella-v-wavlm-layer-11-k-100-lmbda-0-clean-10k-40hz-c4686bf9.pt",
    (0,600):     release_url + "ella-v-wavlm-layer-11-k-100-lmbda-600-clean-10k-40hz-1448a9ec.pt",
    (0,1500):    release_url + "ella-v-wavlm-layer-11-k-100-lmbda-1500-clean-10k-40hz-6bdae6f8.pt",
    (0,3000):    release_url + "ella-v-wavlm-layer-11-k-100-lmbda-3000-clean-10k-40hz-6e277bbd.pt",
    (0,5000):    release_url + "ella-v-wavlm-layer-11-k-100-lmbda-5000-clean-10k-40hz-90149573.pt",
    (0,9000):    release_url + "ella-v-wavlm-layer-11-k-100-lmbda-9000-clean-10k-40hz-58352115.pt",
    (200,0):     release_url + "ella-v-wavlm-layer-11-k-200-lmbda-0-clean-10k-40hz-bc26448e.pt",
    (200,700):   release_url + "ella-v-wavlm-layer-11-k-200-lmbda-700-clean-10k-40hz-32e22dc4.pt",
    (200,1500):  release_url + "ella-v-wavlm-layer-11-k-200-lmbda-1500-clean-10k-40hz-7b009e62.pt",
    (200,3000):  release_url + "ella-v-wavlm-layer-11-k-200-lmbda-3000-clean-10k-40hz-8c19612d.pt",
    (200,5000):  release_url + "ella-v-wavlm-layer-11-k-200-lmbda-5000-clean-10k-40hz-a7d6e988.pt",
    (200,7500):  release_url + "ella-v-wavlm-layer-11-k-200-lmbda-7500-clean-10k-40hz-c6673f9e.pt",
    (500,0):     release_url + "ella-v-wavlm-layer-11-k-500-lmbda-0-clean-10k-40hz-e6f3c91d.pt",
    (500,600):   release_url + "ella-v-wavlm-layer-11-k-500-lmbda-600-clean-10k-40hz-9e639ac6.pt",
    (500,1500):  release_url + "ella-v-wavlm-layer-11-k-500-lmbda-1500-clean-10k-40hz-89352a68.pt",
    (500,2800):  release_url + "ella-v-wavlm-layer-11-k-500-lmbda-2800-clean-10k-40hz-f76d2db5.pt",
    (500,4500):  release_url + "ella-v-wavlm-layer-11-k-500-lmbda-4500-clean-10k-40hz-3815b3ab.pt",
    (500,7000):  release_url + "ella-v-wavlm-layer-11-k-500-lmbda-7000-clean-10k-40hz-f0b78c0f.pt",
    (1000,0):    release_url + "ella-v-wavlm-layer-11-k-1000-lmbda-0-clean-10k-40hz-96650d8d.pt",
    (1000,600):  release_url + "ella-v-wavlm-layer-11-k-1000-lmbda-600-clean-10k-40hz-1505a398.pt",
    (1000,1400): release_url + "ella-v-wavlm-layer-11-k-1000-lmbda-1400-clean-10k-40hz-98bd66a9.pt",
    (1000,2500): release_url + "ella-v-wavlm-layer-11-k-1000-lmbda-2500-clean-10k-40hz-fedea4c6.pt",
    (1000,3800): release_url + "ella-v-wavlm-layer-11-k-1000-lmbda-3800-clean-10k-40hz-d286d597.pt",
    (1000,6000): release_url + "ella-v-wavlm-layer-11-k-1000-lmbda-6000-clean-10k-40hz-5fb405fc.pt",
}
# fmt: on


def _ellav_from_url(
    checkpoint_url: str, map_location="cpu", progress=True
) -> ELLAVGARModel:
    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url,
        map_location=map_location,
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    model_args = TransformerModelArgs.from_dict(checkpoint["model_args"])
    tokenizer = ELLAVTokenizer.from_dict(checkpoint["tokenizer"])
    model = ELLAVGARModel(model_args=model_args, tokenizer=tokenizer, train=False)
    model.transformer.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ELLA-V GAR model with {num_params:,} parameters.")
    return model


def ellav_units_to_wavtokenizer(
    k: int, lmbda: int, map_location="cpu", progress=True
) -> ELLAVGARModel:
    if (k, lmbda) not in ellav_unit_to_wavtokenizer_urls:
        msg = f"Pretrained ELLA-V GAR model for (k, lmbda) = ({k}, {lmbda}) not found. Available models: {ellav_unit_to_wavtokenizer_urls.keys()}"
        raise ValueError(msg)
    checkpoint_url = ellav_unit_to_wavtokenizer_urls[(k, lmbda)]
    return _ellav_from_url(checkpoint_url, map_location=map_location, progress=progress)


def ellav_phones_to_wavtokenizer(map_location="cpu", progress=True) -> ELLAVGARModel:
    return _ellav_from_url(
        ellav_phone_to_wavtokenizer_url, map_location=map_location, progress=progress
    )
