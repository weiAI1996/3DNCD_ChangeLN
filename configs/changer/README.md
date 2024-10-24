# Changer

[3-D Neighborhood Cross-Differencing: A New Paradigm Serves Remote Sensing Change Detection](https://ieeexplore.ieee.org/document/10580934)

## Introduction

## Abstract
Change detection is a prevalent technique in remote sensing image analysis for investigating geomorphological evolution. The modeling and analysis of difference features are crucial for the precise detection of land cover changes. In order to extract difference features, previous work has either directly computed them through differential operations or implicitly modeled them via feature fusion. However, these rudimentary strategies rely heavily on a high degree of congruence within the bitemporal feature space, which results in the model’s diminished capacity to capture subtle variations induced by factors such as differences in illumination. In response to this challenge, the concept of 3-D neighborhood difference convolution (3D-NDC) is proposed for robustly aggregating the intensity and gradient information of features. Furthermore, to delve into the deep disparities within bitemporal instance features, we propose a novel paradigm for differential feature extraction based on 3D-NDC, termed 3-D neighborhood cross-differencing. This strategy is dedicated to exploring the interplay of cross-temporal features, thereby unveiling the inherent disparities among various land cover characteristics. In addition, a detail-focused refinement (DfR) decode based on the Laplace operator has been designed to synergize with the 3-D neighborhood cross-differencing, aiming to improve the detail performance of change instances. This integration forms the basis of a new change detection framework, named ChangeLN. Extensive experiments demonstrate that ChangeLN significantly outperforms other state-of-the-art change detection methods. Moreover, the 3-D neighborhood cross-difference strategy exhibits the potential for integration into other change detection frameworks to improve detection performance.

```bibtex
@ARTICLE{ChangeLN,
  author={Jing, Wei and Chi, Kaichen and Li, Qiang and Wang, Qi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={3-D Neighborhood Cross-Differencing: A New Paradigm Serves Remote Sensing Change Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-11},
  keywords={Feature extraction;Three-dimensional displays;Convolution;Decoding;Spatiotemporal phenomena;Semantics;Task analysis;3-Dneighborhood difference convolution (3D-NDC);change detection;cross-differencing;deep learning;detail-focused refinement (DfR);remote sensing image},
  doi={10.1109/TGRS.2024.3422210}}
```

## Results and models

### LEVIR-CD

|  Method   | Backbone | Crop Size | Lr schd | Mem (GB) | Precision | Recall | F1-Score |  IoU  |  
| :-------: | :------: | :-------: | :-----: | :------: | :-------: | :----: | :------: | :---: |  
| ChangeLN |   r18    |  512x512  |  40000  |    -     |   93.14   | 90.83  |  91.97   | 85.14 || 
| ChangeLN |  MIT-B0  |  512x512  |  40000  |    -     |   93.34   | 90.32  |  91.81   | 84.85 || 


- All metrics are based on the category "change".
- All scores are computed on the test set.
