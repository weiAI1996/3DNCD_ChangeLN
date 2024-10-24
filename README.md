
## Introduction
This is a repository about the paper "3D Neighborhood Cross Differencing: A New Paradigm serves Remote Sensing Change Detection"[[paper]](https://ieeexplore.ieee.org/document/10580934). This repository is based on OPEN-CD (https://github.com/likyoo/open-cd) secondary development. Open-CD is an open source change detection toolbox based on a series of open source general vision task tools. 


## News

## Benchmark and model zoo

Supported toolboxes:

- [x] [OpenMMLab Toolkits](https://github.com/open-mmlab)
- [x] [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [ ] ...

Supported change detection model:
(_The code of some models are borrowed directly from their official repositories._)

- [x] [FC-EF (ICIP'2018)](configs/fcsn)
- [x] [FC-Siam-diff (ICIP'2018)](configs/fcsn)
- [x] [FC-Siam-conc (ICIP'2018)](configs/fcsn)
- [x] [STANet (RS'2020)](configs/stanet)
- [x] [IFN (ISPRS'2020)](configs/ifn)
- [x] [SNUNet (GRSL'2021)](configs/snunet)
- [x] [BiT (TGRS'2021)](configs/bit)
- [x] [ChangeFormer (IGARSS'22)](configs/changeformer)
- [x] [TinyCD (NCA'2023)](configs/tinycd)
- [x] [Changer (TGRS'2023)](configs/changer)
- [x] [HANet (JSTARS'2023)](configs/hanet)
- [x] [TinyCDv2 (Under Review)](configs/tinycd_v2)
- [x] [ChangeLN (TGRS'2024)](configs/changeLN)
- [ ] ...

Supported datasets: | [Descriptions](https://github.com/wenhwu/awesome-remote-sensing-change-detection)
- [x] [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [x] [S2Looking](https://github.com/S2Looking/Dataset)
- [x] [SVCD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
- [x] [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)
- [x] [CLCD](https://github.com/liumency/CropLand-CD)
- [x] [RSIPAC](https://engine.piesat.cn/ai/autolearning/index.html#/dataset/detail?key=8f6c7645-e60f-42ce-9af3-2c66e95cfa27)
- [x] [SECOND](http://www.captain-whu.com/PROJECT/)
- [x] [Landsat](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)
- [x] [BANDON](https://github.com/fitzpchao/BANDON)
- [ ] ...

## Usage

[Docs](https://github.com/open-mmlab/mmsegmentation/tree/master/docs)

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) in mmseg.

A Colab tutorial is also provided. You may directly run on [Colab](https://colab.research.google.com/drive/1puZY5R8fwlL6um6pHbgbM1NTYZUXdK2J?usp=sharing). (thanks to [@Agustin](https://github.com/AgustinNormand) for this demo) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1puZY5R8fwlL6um6pHbgbM1NTYZUXdK2J?usp=sharing)

#### simple usage
```
# Install OpenMMLab Toolkits as Python packages
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpretrain>=1.0.0rc7"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
```
```
git clone https://github.com/weiAI1996/3DNCD_ChangeLN
cd 3DNCD_ChangeLN
pip install -v -e .
```
train
```
python tools/train.py configs/changeLN/changeLN_r18_512x512_40k_levircd.py --work-dir ./changer_r18_levir_workdir
```
infer
```
# get .png results
python tools/test.py configs/changeLN/changeLN_r18_512x512_40k_levircd.py  latest.pth --show-dir tmp_infer
# get metrics
python tools/test.py configs/changeLN/changeLN_r18_512x512_40k_levircd.py  latest.pth
```

## Citation

If you find this project useful in your research, please consider cite:

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

## License

Open-CD is released under the Apache 2.0 license.
