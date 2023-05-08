# GOLF
## Global and Local Hierarchy-aware Contrastive Framework for Hierarchical Implicit Discourse Relation Recognition (ACL 2023)

### Requirements

Firstly, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.10.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.10.1` should also work. 

Secondly, install torch_geometric by following the instructions from [the official website](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Then run the following script to install the remaining dependencies,

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
@misc{jiang2022global,
      title={Global and Local Hierarchy-aware Contrastive Framework for Implicit Discourse Relation Recognition}, 
      author={Yuxin Jiang and Linhan Zhang and Wei Wang},
      year={2022},
      eprint={2211.13873},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
