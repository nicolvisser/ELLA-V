dependencies = ["torch", "torchaudio"]

import torch

from ellav.model import ELLAVGARModel

release_url = (
    "https://github.com/nicolvisser/ella-v-acoustic-model/releases/download/v0.1.0/"
)

# fmt: off
ellav_uam_urls = {
    (0, 0):       release_url + "wavlm-layer-11-km-100-lmbda-0-train-clean-10k-40hz.pt",
    (0, 600):     release_url + "wavlm-layer-11-km-100-lmbda-600-train-clean-10k-40hz.pt",
    (0, 1500):    release_url + "wavlm-layer-11-km-100-lmbda-1500-train-clean-10k-40hz.pt",
    (0, 3000):    release_url + "wavlm-layer-11-km-100-lmbda-3000-train-clean-10k-40hz.pt",
    (0, 5000):    release_url + "wavlm-layer-11-km-100-lmbda-5000-train-clean-10k-40hz.pt",
    (0, 9000):    release_url + "wavlm-layer-11-km-100-lmbda-9000-train-clean-10k-40hz.pt",
    (200, 0):     release_url + "wavlm-layer-11-km-200-lmbda-0-train-clean-10k-40hz.pt",
    (200, 700):   release_url + "wavlm-layer-11-km-200-lmbda-700-train-clean-10k-40hz.pt",
    (200, 1500):  release_url + "wavlm-layer-11-km-200-lmbda-1500-train-clean-10k-40hz.pt",
    (200, 3000):  release_url + "wavlm-layer-11-km-200-lmbda-3000-train-clean-10k-40hz.pt",
    (200, 5000):  release_url + "wavlm-layer-11-km-200-lmbda-5000-train-clean-10k-40hz.pt",
    (200, 7500):  release_url + "wavlm-layer-11-km-200-lmbda-7500-train-clean-10k-40hz.pt",
    (500, 0):     release_url + "wavlm-layer-11-km-500-lmbda-0-train-clean-10k-40hz.pt",
    (500, 600):   release_url + "wavlm-layer-11-km-500-lmbda-600-train-clean-10k-40hz.pt",
    (500, 1500):  release_url + "wavlm-layer-11-km-500-lmbda-1500-train-clean-10k-40hz.pt",
    (500, 2800):  release_url + "wavlm-layer-11-km-500-lmbda-2800-train-clean-10k-40hz.pt",
    (500, 4500):  release_url + "wavlm-layer-11-km-500-lmbda-4500-train-clean-10k-40hz.pt",
    (500, 7000):  release_url + "wavlm-layer-11-km-500-lmbda-7000-train-clean-10k-40hz.pt",
    (1000, 0):    release_url + "wavlm-layer-11-km-1000-lmbda-0-train-clean-10k-40hz.pt",
    (1000, 600):  release_url + "wavlm-layer-11-km-1000-lmbda-600-train-clean-10k-40hz.pt",
    (1000, 1400): release_url + "wavlm-layer-11-km-1000-lmbda-1400-train-clean-10k-40hz.pt",
    (1000, 2500): release_url + "wavlm-layer-11-km-1000-lmbda-2500-train-clean-10k-40hz.pt",
    (1000, 3800): release_url + "wavlm-layer-11-km-1000-lmbda-3800-train-clean-10k-40hz.pt",
    (1000, 6000): release_url + "wavlm-layer-11-km-1000-lmbda-6000-train-clean-10k-40hz.pt",
}
# fmt: on


def ellav_acoustic_model(
    k: int, lmbda: int, map_location="cpu", progress=True
) -> ELLAVGARModel:
    if (k, lmbda) not in ellav_uam_urls:
        raise ValueError(
            f"Pretrained ELLA-V GAR model for (k, lmbda) = ({k}, {lmbda}) not found. Available models: {ellav_uam_urls.keys()}"
        )
    checkpoint_path = ellav_uam_urls[(k, lmbda)]

    checkpoint = torch.load(
        checkpoint_path, weights_only=True, map_location=map_location
    )
    model_args = TransformerInferenceModelArgs.from_dict(checkpoint["model_args"])
    tokenizer = ELLAVTokenizer.from_dict(checkpoint["tokenizer"])
    model = cls(model_args=model_args, tokenizer=tokenizer, train=False)
    model.transformer.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ELLA-V GAR model with {num_params:,} parameters.")
    return model


def codebook(layer: int, k: int, map_location="cpu", progress=True) -> torch.Tensor:
    if (layer, k) not in codebook_urls:
        raise ValueError(
            f"Pretrained codebook for layer {layer} and k {k} not found. Available codebooks: {codebook_urls.keys()}"
        )
    state_dict = torch.hub.load_state_dict_from_url(
        codebook_urls[(layer, k)],
        map_location=map_location,
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    codebook = state_dict["codebook"]
    print(f"WavLM codebook loaded with shape: {codebook.shape}")
    return codebook
