# ICPD-Net

Source code for ICPD-Net, a method that leverages Vision-Language Models (VLMs) to extract cross-modal priors for mitigating intra-class variance issues in remote sensing, thereby enhancing Few-Shot Segmentation (FSS) in remote sensing applications.

## Requirements
- Python 3.9
- PyTorch 2.1
- open-clip-torch

## Model Weights
Download the pretrained ​**RemoteCLIP-ViT-L-14** model weights from:  
https://huggingface.co/chendelong/RemoteCLIP/tree/main

## Dataset Preparation
### Original Dataset Download Links
- ​**iSAID-5i**:  
  https://github.com/caoql98/SDM
- ​**DLRSD-5i**:  
  https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0
### **Softmax-GradCAM** of iSAID-5i for comparison
  https://pan.baidu.com/s/1hXirGlwoAm77DkV_ZpyWPw?pwd=rgig
