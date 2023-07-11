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
    abstract = "Due to the absence of explicit connectives, implicit discourse relation recognition (IDRR) remains a challenging task in discourse analysis. The critical step for IDRR is to learn high-quality discourse relation representations between two arguments. Recent methods tend to integrate the whole hierarchical information of senses into discourse relation representations for multi-level sense recognition. Nevertheless, they insufficiently incorporate the static hierarchical structure containing all senses (defined as global hierarchy), and ignore the hierarchical sense label sequence corresponding to each instance (defined as local hierarchy). For the purpose of sufficiently exploiting global and local hierarchies of senses to learn better discourse relation representations, we propose a novel GlObal and Local Hierarchy-aware Contrastive Framework (GOLF), to model two kinds of hierarchies with the aid of multi-task learning and contrastive learning. Experimental results on PDTB 2.0 and PDTB 3.0 datasets demonstrate that our method remarkably outperforms current state-of-the-art models at all hierarchical levels.",
}
```
