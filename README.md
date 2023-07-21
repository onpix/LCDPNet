<div align="center">
  <a href="https://whyy.site/paper/lcdp">
    <img src="imgs/title.webp"/>
  </a>


  <a href="https://paperswithcode.com/sota/image-enhancement-on-exposure-errors?p=local-color-distributions-prior-for-image">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/local-color-distributions-prior-for-image/image-enhancement-on-exposure-errors" alt="PWC" />
    </a>

  [`🌐 Website`](https://whyy.site/paper/lcdp) &nbsp;&centerdot;&nbsp; [`📃 Paper`](https://www.cs.cityu.edu.hk/~rynson/papers/eccv22b.pdf) &nbsp;&centerdot;&nbsp; [`🗃️ Dataset`](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS?usp=sharing)
</div>


**Abstract:** Existing image enhancement methods are typically designed to address either the over- or under-exposure problem in the input image. When the illumination of the input image contains both over- and under-exposure problems, these existing methods may not work well. We observe from the image statistics that the local color distributions (LCDs) of an image suffering from both problems tend to vary across different regions of the image, depending on the local illuminations. Based on this observation, we propose in this paper to exploit these LCDs as a prior for locating and enhancing the two types of regions (i.e., over-/under-exposed regions). First, we leverage the LCDs to represent these regions, and propose a novel local color distribution embedded (LCDE) module to formulate LCDs in multi-scales to model the correlations across different regions. Second, we propose a dual-illumination learning mechanism to enhance the two types of regions. Third, we construct a new dataset to facilitate the learning process, by following the camera image signal processing (ISP) pipeline to render standard RGB images with both under-/over-exposures from raw data. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art methods quantitatively and qualitatively.

## 📻 News

- 2023.7.21: if you have an interest in low-light enhancement and NeRF, please check out my latest ICCV2023 work, [LLNeRF](https://github.com/onpix/LLNeRF) ! 🔥🔥🔥
- 2023.7.21: Update README
- 2023.2.7: Merge `tar.gz` files of our dataset to a single `7z` file.
- 2023.2.8: Update packages version in `requirements.txt`.
- 2023.2.8: Upload `env.yaml`.

## 🔥 Our Model

![Our model](https://hywang99.github.io/images/lcdpnet/arch.png)


## ⚙️ Setup

1. Clone `git clone https://github.com/onpix/LCDPNet.git`
2. Go to directory `cd LCDPNet`
3. Install required packages `pip install -r requirements.txt`

We also provide `env.yaml` for quickly installing packages. Note that you may need to modify the env name to prevent overwriting your existing enviroment, or modify cudatoolkit and cudnn version in `env.yaml` to match your local cuda version.

## ⌨️ How to run

To train our model:

1. Prepare data: Modify `src/config/ds/train.yaml` and `src/config/ds/valid.yaml`.
2. Modify configs in `src/config`. Note that we use `hydra` for config management.
3. Run: `python src/train.py name=<experiment_name> num_epoch=200 log_every=2000 valid_every=20`

To test our model:

1. Prepare data: Modify `src/config/ds/test.yaml`
2. Run: `python src/test.py checkpoint_path=<file_path>`

## 📂 Dataset & Pretrained Model

The LCDP Dataset is here: [[Google drive]](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS?usp=sharing). Please unzip `lcdp_dataset.7z`. The training and test images are:

|       | Train         | Test               |
| ----- | ------------- | ------------------ |
| Input | `input/*.png` | `test-input/*.png` |
| GT    | `gt/*.png`    | `test-gt/*.png`    |

We provide the two pretrained models: `pretrained_models/trained_on_ours.ckpt` and `pretrained_models/trained_on_MSEC.ckpt` for researchers to reproduce the results in Table 1 and Table 2 in our paper. Note that we train `pretrained_models/trained_on_MSEC.ckpt` on the Expert C subset of the MSEC dataset with both over and under-exposed images.

| Filename             | Training data                                                | Testing data                 | Test PSNR | Test SSIM |
| -------------------- | ------------------------------------------------------------ | ---------------------------- | --------- | --------- |
| trained_on_ours.ckpt | Ours                                                         | Our testing data             | 23.239    | 0.842     |
| trained_on_MSEC.ckpt | [MSEC](https://github.com/mahmoudnafifi/Exposure_Correction) | MSEC testing data (Expert C) | 22.295    | 0.855     |

Our model is lightweight. Experiments show that increasing model size will further improve the quality of the results. To train a bigger model, increase the values in `runtime.bilateral_upsample_net.hist_unet.channel_nums`.

## 🔗 Cite This Paper

If you find our work or code helpful, or your research benefits from this repo, please cite our paper:

```bibtex
@inproceedings{wang2022lcdp,
    title =        {Local Color Distributions Prior for Image Enhancement},
    author =       {Haoyuan Wang, Ke Xu, and Rynson W.H. Lau},
    booktitle =    {Proceedings of the European Conference on Computer Vision (ECCV)},
    year =         {2022}
}
```