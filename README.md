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
## Model Weights for iSAID-5i and DLRSD-5i  
Below are the weights for 1-shot and 5-shot experiments on the iSAID-5i and DLRSD-5i datasets. All files are hosted via Baidu Netdisk.

---

### iSAID-5i  
#### 1-Shot iSAID-5i  
| Fold | File Name                    | Baidu Netdisk Link                                                                 |
|------|------------------------------|------------------------------------------------------------------------------------|
| 0    | `1_shot_0_fold_ISAID.pt`     | [Download](https://pan.baidu.com/s/1ToERONYuVy6syKCUYX_ukQ?pwd=qgy8). (Extract: qgy8) |
| 1    | `1_shot_1_fold_ISAID.pt`     | [Download](https://pan.baidu.com/s/1gOo_Hr1vP7MTS6hdFMfcdg?pwd=gth9) (Extract: gth9) |
| 2    | `1_shot_2_fold_ISAID.pt`     | [Download](https://pan.baidu.com/s/1TK4feEi6NaunlE6P1bFYZQ?pwd=r8jj) (Extract: r8jj) |

#### 5-Shot iSAID-5i  
| Fold | File Name                    | Baidu Netdisk Link                                                                 |
|------|------------------------------|------------------------------------------------------------------------------------|
| 0    | `5_shot_0_fold_ISAID.pt`     | [Download](https://pan.baidu.com/s/1wp8gZpT01JkTiKt51TWRyA?pwd=tj9d) (Extract: tj9d) |
| 1    | `5_shot_1_fold_ISAID.pt`     | [Download](https://pan.baidu.com/s/1gfQIA4qID60PUPFB_zSx4A?pwd=h4jg) (Extract: h4jg) |
| 2    | `5_shot_2_fold_ISAID.pt`     | [Download](https://pan.baidu.com/s/1ZDTDWGGxOO2_L74psTrXHw?pwd=nv5x) (Extract: nv5x) |

---

### DLRSD-5i  
#### 1-Shot DLRSD-5i  
| Fold | File Name                    | Baidu Netdisk Link                                                                 |
|------|------------------------------|------------------------------------------------------------------------------------|
| 0    | `1_shot_0_fold_DLRSD.pt`     | [Download](https://pan.baidu.com/s/1rd9ahHgt8gtC2IsWqb7k4A?pwd=29xd) (Extract: 29xd) |
| 1    | `1_shot_1_fold_DLRSD.pt`     | [Download](https://pan.baidu.com/s/13pedxzQQjFrFGOj1JjGLSA?pwd=7r62) (Extract: 7r62) |
| 2    | `1_shot_2_fold_DLRSD.pt`     | [Download](https://pan.baidu.com/s/1P1MTMdy7FIi6K35xjqjNnw?pwd=erf8) (Extract: erf8) |

#### 5-Shot DLRSD-5i  
| Fold | File Name                    | Baidu Netdisk Link                                                                 |
|------|------------------------------|------------------------------------------------------------------------------------|
| 0    | `5_shot_0_fold_DLRSD.pt`     | [Download](https://pan.baidu.com/s/198oqwwhuhpSfGbTERdcytg?pwd=wk9d) (Extract: wk9d) |
| 1    | `5_shot_1_fold_DLRSD.pt`     | [Download](https://pan.baidu.com/s/1j-sYTY7045y5VNRk9LHugw?pwd=2ng7) (Extract: 2ng7) |
| 2    | `5_shot_2_fold_DLRSD.pt`     | [Download](https://pan.baidu.com/s/1kOHeQPY3TuXutrlsxNntLw?pwd=65t4) (Extract: 65t4) |



## Acknowledgement
This repo benefits from [PCFNet](https://github.com/TinyAway/PCFNet), [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP), and [CLIP](https://github.com/openai/CLIP). Thanks for their works.
