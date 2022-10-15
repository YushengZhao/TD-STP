# Target-Driven Structured Transformer Planner for Vision-Language Navigation

This is the official implementation of **MM'22** (ACM International Conference on Multimedia) paper (accepted as **oral**): Target-Driven Structured Transformer Planner for Vision-Language Navigation.

## Citation

```
@inproceedings{zhao2022target,
  title={Target-Driven Structured Transformer Planner for Vision-Language Navigation},
  author={Zhao, Yusheng and Chen, Jinyu and Gao, Chen and Wang, Wenguan and Yang, Lirong and Ren, Haibing and Xia, Huaxia and Liu, Si},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4194--4203},
  year={2022}
}
```

## Installation

Please refer to [this repo](https://github.com/cshizhe/VLN-HAMT) for installation guides.

## Training & Inference

To train on R2R:
```shell
cd finetune_src
python ./scripts/run_r2r.sh
```

To train on REVERIE:
```shell
cd finetune_src
python ./scripts/run_reverie.sh
```

To test on R2R:
```shell
cd finetune_src
python ./scripts/test_r2r.sh
```
To train on REVERIE:
```shell
cd finetune_src
python ./scripts/test_r2r.sh
```

Note that some file paths in the code may require slight adaptation.

## Acknowledgement

This code is based on [HAMT](https://github.com/cshizhe/VLN-HAMT). We appreciate their great contribution to the community.
