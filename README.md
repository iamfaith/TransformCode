# TransformCode

Code for the journal paper: **TransformCode: A Contrastive Learning Framework for Code Embedding via Subtree Transformation**

## Code Transformation

All code for code transformation is under the folder of **code_transformation**. It needs the support of tree-sitter.zip.

1. unzip tree-sitter.zip to **$HOME** dir.
2. All the methods can be imported from __init__.py

## TransformCode

1. Install python dependency： pip install aicmder transformer commode_utils
2. Our TransformCode model is CodeEmbed, located in method_name_prediction/model.py.

### method name prediction

We have uploaded all codes, weights and data to reproduce the experiments of method name prediction.

1. change the dataset section in method_name_prediction/config.yaml (Optional: if you run from project folder, no needed)
2. After installing the dependency, run: python run.py

#### CodeBERT

method name prediction with CodeBERT

The model file is codebert_predictor.py

Weight can be downloaded from here (Github lfs space is limited to 1Gb, so we use netdisk):
链接：https://pan.baidu.com/s/1IMaBapXZ6_tXSdxYMbQIdg?pwd=csci 
提取码：csci