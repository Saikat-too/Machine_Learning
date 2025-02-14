# **Machine Learning/Deep Learning Implementation**

This repository consists implementations of various machine learning and deep learning papers , algorithms and architecture from scratch.

# **Goals**
## **Classical ML Implementations** <br>
1. Implement SGD for logistic regression from scratch.<br>
2. Compare Ridge vs Lass regression on a synthetic dataset.<br>
3. Implement EM for Gaussian Mixture Models(use scipy.stats).<br>
4. Build a random forest from scratch (use sklearn.tree as a reference).<br>
5. Code a kernel SVM (use quadratic programming with cvxopt).<br>
## **Neural Networks and Deep Learning**
6. Build a 2-layer MLP with backprop from scratch(NumPy only).<br>
7. Reimplement AlexNet in pytorch (simplify layers).<br>
8. Train an LSTM on text generation.<br>
9. Implement a transoformer block for masked language modeling.<br>
10.Compare SGD vs Adam on CNN(track loss curves).<br>
## **Generative Models**
11.Train a DCGAN on CIFAR-10(use pytorch).<br>
12.Implement SimCLR on MNIST with PyTorch Lightning.<br>
13.Fine-tune BERT on a custome dataset.<br>
14.Prune a pre-trained ResNet and measure accuracy drop.<br>
15.Integrate SAM into a ResNet training loop (PyTorch). Compare convergence vs SGD/Adam.<br>
## **Advanced Deep Learning**
16.Compare NTK for a 3-layer MLP and visualize its evolution during training.<br>
17.Implement MoE layer with top-2 routing(use PyTorch). Test on a multilingual translation task.<br>
18.Train a SIREN to represent an image or audio signal (no grids!).<br>
19.Implement a simplified DARTS workflow to search for CNN cells on CIFAR-10.<br>
20.Train a diffusion model on MNIST with PyTorch.<br>
## **Advanced Architecture**
21.Build a Glow inspired flow model for image generation.<br>
22.Implement Bayesian layers with Monte Carlo dropout.<br>
23.Train an ensemble of 5 ResNets and measure uncertainty on out-of-distribution data.<br>
24.Train an EBM to generate CIFAR-10 samples usin Langevin dynamics.<br>
25.Fine tune GPT -2 (of LLaMA) on a custom dataset using LoRA(HuggingFace + PyTorch).<br>
## **LLM and Production**
26.Implement Chain of Thought with GPT-3/Claude for math word problems.<br>
27.Build a RAG system with FAISS(vector DB) with T5 for QA.<br>
28.Implement local/stride attention patterns for text generation.<br>
29.Train a transformer to discover symbolic equation from data(e.g , F = ma).<br>
30.Train a ResNet with FP16/AMP and benchmark speed vs FP32.<br>
## **System Design and Optimization**
31.Split a transformer across multiple GPUs(Pytorch nn.parallel).<br>
32.Quantize a Vit to 8-bit and measure accuracy drop.<br>
33.Implement FlashAttention for a transformer blcok(CUDA optional).<br>
34.Train a large model (e.g., GPT-2 XL) using PyTorch FSDP.<br>
35.Train a NeRF on a custom 3D scene (use PyTorch3D or Nerfstudio).<br>
## **Application**
36.Implement PPO for RLHF on a text summarization task.<br>
37.Train a GAT for node classification on Cora/MolHIV.<br>
38.Build a world model for CartPole control using latent imagination.<br>
39.Reproduce a simplified protein folding pipeline(use PyRosetta).<br>



Cya........
