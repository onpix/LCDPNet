# Local Color Distributions Prior for Image Enhancement [ECCV2022]

Haoyuan Wang<sup>1</sup>, Ke Xu<sup>1</sup>, Rynson Lau<sup>1</sup>

<sup>1</sup>City University of Hong Kong

[[ Project page ]](https://hywang99.github.io/2022/07/09/lcdpnet/)
[[ Dataset & Pretrained models ]](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS?usp=sharing)


| Input | Ours                                                                                   |
|-------|----------------------------------------------------------------------------------------|
| ![](https://hywang99.github.io/images/lcdpnet/res0-a1273-IMG_1444.png) | ![](https://hywang99.github.io/images/lcdpnet/res1-a1273-IMG_1444.png)                 |
|  ![](https://hywang99.github.io/images/lcdpnet/res0-a0259-dvf_029.png)     | ![](https://hywang99.github.io/images/lcdpnet/res1-a0259-dvf_029.png)                  |

[//]: # (|  ![]&#40;https://hywang99.github.io/images/lcdpnet/res0-a1682-DSC_0010-4.png&#41;     | ![]&#40;https://hywang99.github.io/images/lcdpnet/res1-a1682-DSC_0010-4.png&#41;               |)

[//]: # (|  ![]&#40;https://hywang99.github.io/images/lcdpnet/res0-a2117-20050510_213735__MG_1270.png&#41;     | ![]&#40;https://hywang99.github.io/images/lcdpnet/res1-a2117-20050510_213735__MG_1270.png&#41; |)

## Abstract 

Existing image enhancement methods are typically designed to address either the over- or under-exposure problem in the input image. When the illumination of the input image contains both over- and under-exposure problems, these existing methods may not work well. We observe from the image statistics that the local color distributions (LCDs) of an image suffering from both problems tend to vary across different regions of the image, depending on the local illuminations. Based on this observation, we propose in this paper to exploit these LCDs as a prior for locating and enhancing the two types of regions (i.e., over-/under-exposed regions). First, we leverage the LCDs to represent these regions, and propose a novel local color distribution embedded (LCDE) module to formulate LCDs in multi-scales to model the correlations across different regions. Second, we propose a dual-illumination learning mechanism to enhance the two types of regions. Third, we construct a new dataset to facilitate the learning process, by following the camera image signal processing (ISP) pipeline to render standard RGB images with both under-/over-exposures from raw data. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art methods quantitatively and qualitatively. Codes and dataset are in https://hywang99.github.io/lcdpnet/.

![Our model.](https://hywang99.github.io/images/lcdpnet/arch.png)

## Running

To train our model:

1. Prepare data: Modify `src/config/ds/train.yaml` and `src/config/ds/valid.yaml`.
2. Modify configs in `src/config`. Note that we use `hydra` for config management.
3. Run: `python src/train.py name=<experiment_name> num_epoch=200 log_every=2000 valid_every=20`

To test our model:

1. Prepare data: Modify `src/config/ds/test.yaml`
2. Run: `python src/test.py checkpoint_path=<file_path>`

## Dataset & Pretrained Model

[[ Google drive ]](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS?usp=sharing)

Unzip two `tar.gz` files: `dataset/raise.tar.gz` and `dataset/adobe5k.tar.gz`. The training and test dataset are:

|       | Train | Test |
|-------|-------|------|
| Input | `adobe5k/input/*.png` <br /> `raise/input/*.png` | `adobe5k/test-input/*.png` |
| GT    | `adobe5k/gt/*.png` <br /> `raise/gt/*.png`       | `adobe5k/test-gt/*.png` |

We provide the two pretrained models `pretrained_models/trained_on_ours.ckpt` and `pretrained_models/trained_on_MSEC.ckpt` for researchers to reproduce the results in Table 1. and Table 2. in our paper.

## Contact Me

If you have any question, feel free to open issues or contact me via hywang26-c@my.cityu.edu.hk. (> ω < )
