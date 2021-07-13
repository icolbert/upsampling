# Convolution-based Upsampling - Coded for clarity, not speed.

State-of-the-art deep learning solutions for image upsampling are currently trained using either resize or sub-pixel convolution to learn kernels that generate high fidelity images with minimal artifacts.
However, performing inference with these learned convolution kernels requires memory-intensive feature map transformations that dominate time and energy costs in real-time applications. 
To alleviate this pressure on memory bandwidth, we confine the use of resize or sub-pixel convolution to training in the cloud by transforming learned convolution kernels to deconvolution kernels before deploying them for inference as a functionally equivalent deconvolution.
These kernel transformations, intended as a one-time cost when shifting from training to inference, enable a systems designer to use each algorithm in their optimal context by preserving the image fidelity learned when training in the cloud while minimizing data transfer penalties during inference at the edge.
We explore existing variants of deconvolution inference algorithms and introduce a novel variant for consideration, the improved reverse looping deconvolution algorithm (REVD2).
