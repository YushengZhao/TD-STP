# Target-Driven Structured Transformer Planner for Vision-Language Navigation

This is the official implementation of MM'22 paper (accepted as oral): Target-Driven Structured Transformer Planner for Vision-Language Navigation.

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

## Citation

```
@article{zhao2022target,
  title={Target-Driven Structured Transformer Planner for Vision-Language Navigation},
  author={Zhao, Yusheng and Chen, Jinyu and Gao, Chen and Wang, Wenguan and Yang, Lirong and Ren, Haibing and Xia, Huaxia and Liu, Si},
  journal={arXiv preprint arXiv:2207.11201},
  year={2022}
}
```

## Acknowledgement

This code is based on [HAMT](https://github.com/cshizhe/VLN-HAMT). We appreciate their great contribution to the community.