# ELLA-V with WavTokenizer

This repository contains an **unofficial** implementation of the ELLA-V speech synthesis model from this [paper](https://arxiv.org/abs/2401.07333).

Instead of using a multi-level codec like [Encodec](https://github.com/facebookresearch/encodec), we use the [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) codec with a single level.
Therefore we only need the generalized autoregressive model (GAR) from ELLA-V and not the non-autoregressive model (NAR).

⚠️ Important:
We use ELLA-V as an acoustic model to generate speech from discrete units.
In other words, we use it in a textless-NLP (spoken language modeling) setup and not in a text-to-speech setup.
Therefore, the pretrained checkpoints may be of little interest to you, if you are looking for a text-to-speech model.
See our paper, [Spoken Language Modeling with Duration-Penalized Self-Supervised Units](https://arxiv.org/abs/2505.23494), for more details.
