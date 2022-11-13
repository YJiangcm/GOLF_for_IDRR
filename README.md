# GOLF_for_IDRR
Global and Local Hierarchy-aware Contrastive Framework for Hierarchical Implicit Discourse Relation Recognition

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.10.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.10.1` should also work. 

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Data Preparation before Training

1. Download the [PDTB2.0](https://github.com/cgpotts/pdtb2) dataset, put it under /raw/
2. Run the following script for data preprocessing,
```bash
python3 preprocess.py
```
3. Download the pretrained model [roberta base](https://huggingface.co/roberta-base/tree/main), put it under /pretrained/roberta-base/

### Train, Evaluate, and Test
Run the following script for training, evaludating, and testing,
```bash
sh repeats.sh
```
