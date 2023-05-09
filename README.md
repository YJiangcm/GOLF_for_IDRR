# GOLF
## Global and Local Hierarchy-aware Contrastive Framework for Hierarchical Implicit Discourse Relation Recognition (ACL 2023)

### Requirements

1. Install `PyTorch` by following the instructions from [the official website](https://pytorch.org). 

2. Install `torch_geometric` by following the instructions from [the official website](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

3. Run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Data Preparation before Training

1. Download the [PDTB 2.0](https://github.com/cgpotts/pdtb2) dataset, put it under /raw/
2. Run the following script for data preprocessing,
```bash
python3 preprocess.py
```
(**P.S.** PDTB 3.0 can be downloaded from [https://catalog.ldc.upenn.edu/LDC2019T05](https://catalog.ldc.upenn.edu/LDC2019T05). You can easily modify [preprocess.py](https://github.com/YJiangcm/GOLF_for_IDRR/blob/master/preprocess.py) and adapt it to PDTB 3.0.) 

### Train, Evaluate, and Test
Run the following script for training, evaludating, and testing,
```bash
python3 run.py
```

### Ciation
Please cite our paper by:
```bibtex
@article{DBLP:journals/corr/abs-2211-13873,
  author       = {Yuxin Jiang and
                  Linhan Zhang and
                  Wei Wang},
  title        = {Global and Local Hierarchy-aware Contrastive Framework for Implicit
                  Discourse Relation Recognition},
  journal      = {CoRR},
  volume       = {abs/2211.13873},
  year         = {2022},
  url          = {https://doi.org/10.48550/arXiv.2211.13873},
  doi          = {10.48550/arXiv.2211.13873},
  eprinttype    = {arXiv},
  eprint       = {2211.13873},
  timestamp    = {Tue, 29 Nov 2022 17:41:18 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2211-13873.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
