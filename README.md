# **Recurrent Feature Reasoning for Medical Image Inpainting**
The repository containing the code and results for the CS736 (Medical Image Computing) course at IIT Bombay in Spring 2023.

The paper followed:

```
@INPROCEEDINGS {9156533,
author = {J. Li and N. Wang and L. Zhang and B. Du and D. Tao},
booktitle = {2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
title = {Recurrent Feature Reasoning for Image Inpainting},
year = {2020},
volume = {},
issn = {},
pages = {7757-7765},
abstract = {Existing inpainting methods have achieved promising performance for recovering regular or small image defects. However, filling in large continuous holes remains difficult due to the lack of constraints for the hole center. In this paper, we devise a Recurrent Feature Reasoning (RFR) network which is mainly constructed by a plug-and-play Recurrent Feature Reasoning module and a Knowledge Consistent Attention (KCA) module. Analogous to how humans solve puzzles (i.e., first solve the easier parts and then use the results as additional information to solve difficult parts), the RFR module recurrently infers the hole boundaries of the convolutional feature maps and then uses them as clues for further inference. The module progressively strengthens the constraints for the hole center and the results become explicit. To capture information from distant places in the feature map for RFR, we further develop KCA and incorporate it in RFR. Empirically, we first compare the proposed RFR-Net with existing backbones, demonstrating that RFR-Net is more efficient (e.g., a 4% SSIM improvement for the same model size). We then place the network in the context of the current state-of-the-art, where it exhibits improved performance. The corresponding source code is available at: https://github.com/jingyuanli001/RFR-Inpainting.},
keywords = {cognition;convolution;semantics;merging;convolutional codes;correlation;computational efficiency},
doi = {10.1109/CVPR42600.2020.00778},
url = {https://doi.ieeecomputersociety.org/10.1109/CVPR42600.2020.00778},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {jun}
}
```

Author's repository containing the torch implemetation of the model: https://github.com/jingyuanli001/RFR-Inpainting/

This project was done in a group of two, but the results were obtained indpendently on different datasets, so they are distributed in the folders `ashwin/` and `ravi/` respectively.