# Efficient Convolution-based Image Upsampling
### Coded for clarity, not speed.

State-of-the-art deep learning solutions for image upsampling are currently trained using either resize or sub-pixel convolution to learn kernels that generate high fidelity images with minimal artifacts. However, performaing inference with these learned convolution kernels requires memory-intensive feature map transformations that dominate time and energy costs in real-time applications. In our paper, we introduce kernel transformations that alleviate this pressure on memory bandwidth by transforming learned convolution kernels to deconvolutoin kernels. By confining the use of resize or sub-pixel convolution to training in the cloud where the data transfer penalties are less severe, we minimize these time and energy costs at inference time.

Here, we provide easy-to-understand implementations for all algorithms described in our paper. The code and images used in this repository are free to use as regulated by the license and subject to proper arbitration:

- [[1] Colbert *et al.* (2021) - An Energy-Efficient Edge Computing Paradigm for Convolution-based Image Upsampling](https://arxiv.org/abs/2107.07647)

```
@misc{colbert2021energyefficient,
      title={An Energy-Efficient Edge Computing Paradigm for Convolution-based Image Upsampling}, 
      author={Ian Colbert and Ken Kreutz-Delgado and Srinjoy Das},
      year={2021},
      eprint={2107.07647},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
