<div style="text-align: center">
<h1> Local Color Distributions Prior for Image Enhancement [ECCV2022]
</h1>

Haoyuan Wang<sup>1</sup>, Ke Xu<sup>1</sup>, Rynson Lau<sup>1</sup>

<sup>1</sup>City University of Hong Kong

| Input                                                                                                           | Ours                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| <img width=250px src="https://hywang99.github.io/images/lcdpnet/res0-a2117-20050510_213735__MG_1270.png"></img> | <img width=250px src="https://hywang99.github.io/images/lcdpnet/res1-a2117-20050510_213735__MG_1270.png"></img> |

[//]: # (| ![]&#40;https://hywang99.github.io/images/lcdpnet/res0-a1273-IMG_1444.png&#41; | ![]&#40;https://hywang99.github.io/images/lcdpnet/res1-a1273-IMG_1444.png&#41;                 |)
[//]: # (|  ![]&#40;https://hywang99.github.io/images/lcdpnet/res0-a0259-dvf_029.png&#41;     | ![]&#40;https://hywang99.github.io/images/lcdpnet/res1-a0259-dvf_029.png&#41;                  |)
[//]: # (|  ![]&#40;https://hywang99.github.io/images/lcdpnet/res0-a1682-DSC_0010-4.png&#41;     | ![]&#40;https://hywang99.github.io/images/lcdpnet/res1-a1682-DSC_0010-4.png&#41;               |)

<h2 style="font-size: 2rem; margin-bottom:1rem;">
See more interactive comparisons on our <a href="https://whyy.site/paper/lcdp" style="margin: auto;">[ project page ]</a> !
</h2>
</div>

![](https://github.com/onpix/LCDPNet/blob/main/fig1.jpg)

---

## Abstract 

Existing image enhancement methods are typically designed to address either the over- or under-exposure problem in the input image. When the illumination of the input image contains both over- and under-exposure problems, these existing methods may not work well. We observe from the image statistics that the local color distributions (LCDs) of an image suffering from both problems tend to vary across different regions of the image, depending on the local illuminations. Based on this observation, we propose in this paper to exploit these LCDs as a prior for locating and enhancing the two types of regions (i.e., over-/under-exposed regions). First, we leverage the LCDs to represent these regions, and propose a novel local color distribution embedded (LCDE) module to formulate LCDs in multi-scales to model the correlations across different regions. Second, we propose a dual-illumination learning mechanism to enhance the two types of regions. Third, we construct a new dataset to facilitate the learning process, by following the camera image signal processing (ISP) pipeline to render standard RGB images with both under-/over-exposures from raw data. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art methods quantitatively and qualitatively. Codes and dataset are in https://whyy.site/paper/lcdp.

![Our model.](https://hywang99.github.io/images/lcdpnet/arch.png)

## Setup

1. Clone `git clone https://github.com/onpix/LCDPNet.git`
2. Go to directory `cd LCDPNet`
3. Install required packages `pip install -r requirements.txt`

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

We provide the two pretrained models `pretrained_models/trained_on_ours.ckpt` and `pretrained_models/trained_on_MSEC.ckpt` for researchers to reproduce the results in Table 1. and Table 2. in our paper. Note that we train `pretrained_models/trained_on_MSEC.ckpt` on the Expert C subset of the MSEC dataset with both over and under-exposed images.

|  Filename   | Training data | Testing data | Test PSNR | Test SSIM |  
|-------|-------|------|-----|-----|
| trained_on_ours.ckpt | Ours | Our testing data  | 23.239  |  0.842 |
| trained_on_MSEC.ckpt | [MSEC](https://github.com/mahmoudnafifi/Exposure_Correction)  | MSEC testing data (Expert C)  | 22.295   |  0.855 |

Our model is lightweight. Experiments show that increasing model size will further improve the quality of the results. To train a bigger model, increase the values in `runtime.bilateral_upsample_net.hist_unet.channel_nums`.

## Citation

If you find our work or code helpful, or your research benefits from this repo, please cite our paper:

```
@inproceedings{wang2022lcdp,
    title =        {Local Color Distributions Prior for Image Enhancement},
    author =       {Haoyuan Wang, Ke Xu, and Rynson W.H. Lau},
    booktitle =    {Proceedings of the European Conference on Computer Vision (ECCV)},
    year =         {2022}
}
```

## Contact Me

If you have any question, feel free to open issues or contact me via hywang26-c@my.cityu.edu.hk. (> ω < )
