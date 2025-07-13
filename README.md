## Spiking Transformers Need High Frequency Information

Spiking Transformers Need High Frequency Information: [[Paper]](https://arxiv.org/abs/2505.18608). 
  
### Summary

Spiking Transformers offer an energy-efficient alternative to conventional deep learning by transmitting information solely through binary (0/1) spikes. However, there remains a substantial performance gap compared to artificial neural networks. A common belief is that their binary and sparse activation transmission leads to information loss, thus degrading feature representation and accuracy. In this work, however, we reveal for the first time that spiking neurons preferentially propagate low-frequency information. We hypothesize that the rapid dissipation of high-frequency components is the primary cause of performance degradation. For example, on Cifar-100, adopting Avg-Pooling (low-pass) for token mixing lowers performance to 76.73%; interestingly, replacing it with Max-Pooling (high-pass) pushes the top-1 accuracy to 79.12%, surpassing the well-tuned Spikformer baseline by 0.97%. Accordingly, we introduce Max-Former that restores high-frequency signals through two frequency-enhancing operators: extra Max-Pooling in patch embedding and Depth-Wise Convolution in place of self-attention. Notably, our Max-Former (63.99 M) hits the top-1 accuracy of 82.39% on ImageNet, showing a +7.58% improvement over Spikformer with comparable model size (74.81%, 66.34 M). On cifar10/100, cifar10-dvs we achieved 97.04%/82.65% and 84.2% performance respectively.

This paper also explains why LIF performs better than IF from a frequency domain perspective, that is, LIF neurons can retain more high-frequency information than IF neurons. We hope this simple yet effective solution inspires future research to explore the distinctive nature of spiking neural networks, beyond the established practice in standard deep learning.

### Implementation

This repository include all the patch embeding and token mixing strategies listed in our [[Paper]](https://arxiv.org/abs/2505.18608). Code for QKFormer with membrane shortcut and SSA-DWC which we did not discuss in detail in the paper can be found in ``mixer_hub.py``.

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

Please check the bash file in each folder (cifar10-100, event). 

Currently, this work is partially open sourced.  Code/checkpoints for ImageNet will be available in a few weeks once we finish cleaning up the code.

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
