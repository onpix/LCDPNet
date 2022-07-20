# Local Color Distributions Prior for Image Enhancement [ECCV2022]

Haoyuan Wang<sup>1</sup>, Ke Xu<sup>1</sup>, Rynson Lau<sup>1</sup>

<sup>1</sup>City University of Hong Kong

[[ Project page ]](https://hywang99.github.io/2022/07/09/lcdpnet/)

![](https://hywang99.github.io/images/lcdpnet/arch.png)

## Requirements

- torch == 1.12
- pytorch-lightning == 1.6.5

## Train

1. Prepare data: Modify `src/config/ds/train.yaml` and `src/config/ds/valid.yaml`
2. Modify configs in `src/config`. Note that we use `hydra` for config management.
3. Run: `python src/train.py name=<experiment_name> num_epoch=200 log_every=2000 valid_every=20`

## Test

1. Prepare data: Modify `src/config/ds/test.yaml`
2. Run: `python src/test.py checkpoint_path=<file_path>`

## Dataset

[[ Google drive ]](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS?usp=sharing)

There are two `tar.gz` files: `raise.tar.gz` and `adobe5k.tar.gz`. Unzip them. The training and test dataset are:

|       | Train | Test |
|-------|-------|------|
| Input | `adobe5k/input/*.png` <br /> `raise/input/*.png` | `adobe5k/test-input/*.png` |
| GT    | `adobe5k/gt/*.png` <br /> `raise/gt/*.png`       | `adobe5k/test-gt/*.png` |

## Pretrained model

[to be released]

## Contact

If you have any question, feel free to open issues or contact me via hywang26-c@my.cityu.edu.hk. PR is also welcome ! (> ω < )
