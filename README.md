# Global and Local Hierarchy-aware Contrastive Framework for Hierarchical Implicit Discourse Relation Recognition (ACL 2023)

arXiv preprint: https://arxiv.org/abs/2211.13873

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
（Our code can be easily run on a single NVIDIA GeForce RTX 3090)

### Citation
If you find this work helpful, please cite our paper by:
```bibtex
@inproceedings{jiang-etal-2023-global,
    title = "Global and Local Hierarchy-aware Contrastive Framework for Implicit Discourse Relation Recognition",
    author = "Jiang, Yuxin  and
      Zhang, Linhan  and
      Wang, Wei",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.510",
    pages = "8048--8064",
}
```
