## Spiking Neural Networks Need High-Frequency Information

[Neurips 2025] Spiking Neural Networks Need High-Frequency Information: [[Paper]](https://arxiv.org/abs/2505.18608). [[OpenReview]](https://openreview.net/forum?id=owNPAl7LNK) [[Poster Session]](https://neurips.cc/virtual/2025/loc/san-diego/poster/115997)

**TL;DR: The paper reveals that the performance gap between SNNs and ANNs stems not from information loss caused by binary spike activations, but from the intrinsic low-pass filtering of spiking neurons.**

<img width="1722" height="860" alt="image" src="https://github.com/user-attachments/assets/3d760878-5a70-44e5-b9ee-84a5289b0706" />


### Summary

This paper shows that **spiking neurons are low-pass filters and also explains why LIF performs better than IF from a frequency domain perspective**. That is, LIF neurons can retain more high-frequency information than IF neurons. We hope this simple yet effective solution inspires future research to explore the distinctive nature of spiking neural networks, beyond the established practice in standard deep learning. The core contributions are:

 • We provide the first theoretical proof that spiking neurons inherently act as low-pass filters at the network level, revealing their tendency to suppress high-frequency features.
 
 • We propose **Max-Former**, which restores high-frequency information in Spiking Transformers via two lightweight modules: extra Max-Pool in patch embedding and Depth-Wise Convolution in place of early-stage self-attention.
 
 • Restoring high-frequency information significantly improves performance while saving energy cost. On ImageNet, **Max-Former** achieves **82.39% top-1 accuracy (+7.58% over Spikformer) with 30% energy consumption and lower parameter count (63.99M vs. 66.34M).**
 
 • Extending the insight beyond transformers, **Max-ResNet-18** achieves **state-of-the-art** performance on convolution-based benchmarks: **97.17% on CIFAR-10 and 83.06% on CIFAR-100.**

![](https://github.com/user-attachments/assets/c1e6144b-fe5c-49d3-8d2b-a214ae3e024d)


### Implementation

This repository includes all the patch embedding and token mixing strategies listed in our [[Paper]](https://arxiv.org/abs/2505.18608). Code for token mixing strategies can be found in ``mixer_hub.py``, including SSA-DWC that we did not discuss in detail in the paper. Patch embedding strategies can be found in ``embedding_hub.py``.

#### Requirement:

```bash
  pip install timm==0.6.12 spikingjelly==0.0.0.0.12 opencv-python==4.8.1.78 wandb einops PyYAML Pillow six torch

  ### OPTIONAL 1: apex
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

  ### OPTIONAL 2: cupy
  pip install cupy tensorboard
```

#### Running the code

Please check the bash file in each folder (cifar10-100, event, imagenet).

It can be run directly through the provided `.sh` file. The README will be updated accordingly within this week.



Code for visualization/energy consumption will be uploaded upon request. 




#### Citation

If you find this repo helpful, we’d appreciate it if you cited our work.

```
@article{fang2025spiking,
  title={Spiking Transformers Need High Frequency Information},
  author={Fang, Yuetong and Zhou, Deming and Wang, Ziqing and Ren, Hongwei and Zeng, ZeCui and Li, Lusong and Zhou, Shibo and Xu, Renjing},
  journal={arXiv preprint arXiv:2505.18608},
  year={2025}
}
```
