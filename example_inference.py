import torch
import torchaudio
from IPython.display import Audio, display

wav, sr = torchaudio.load(
    "/mnt/wsl/nvme/datasets/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
)
assert sr == 16000

LAYER = 11
K = 500

wavlm = torch.hub.load(
    "nicolvisser/wavlm-codebooks",
    "wavlm_large",
).cuda()
wavlm.eval()

codebook = torch.hub.load(
    "nicolvisser/wavlm-codebooks",
    "codebook",
    layer=LAYER,
    k=K,
).cuda()

ellav = torch.hub.load(
    "nicolvisser/ELLA-V",
    "ellav_units_to_wavtokenizer",
    k=K,
    lmbda=0,
).cuda()
ellav.eval()

wavtokenizer = torch.hub.load(
    "nicolvisser/WavTokenizer",
    "small_600_24k_4096",
).cuda()
wavtokenizer.eval()

with torch.inference_mode():
    features, _ = wavlm.extract_features(
        source=wav.cuda(),
        padding_mask=None,
        mask=False,
        ret_conv=False,
        output_layer=LAYER,
        ret_layer_results=False,
    )  # [1, T, D]
    features = features.squeeze(0)  # [T, D]

    distances = torch.cdist(features.cuda(), codebook, p=2)  # [T, K]
    units = torch.argmin(distances, dim=1)  # [T,]

    units = [f"u{unit}" for unit in units]

    codec_ids_list, finished = ellav.generate(
        prompts=[units] * 3,
        max_tokens=1000,
        max_codec_tokens_per_phone=10,
        temperature=1.0,
        top_p=0.8,
        chunk_size=None,
        progress=True,
    )

    for codec_ids in codec_ids_list:

        # decode with wavtokenizer
        features = wavtokenizer.codes_to_features(codec_ids[None, None, :].cuda())
        bandwidth_id = torch.tensor([0], device="cuda")
        audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

        display(Audio(audio_out.cpu().numpy(), rate=24000))
