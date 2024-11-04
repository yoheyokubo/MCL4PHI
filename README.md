# Multi-Instance Contrastive Learning with Binomial K-Mers for Phage Host Interaction Prediction

This repository is the official implementation of [Multi-Instance Contrastive Learning with Binomial
K-Mers for Phage Host Interaction Prediction](https://arxiv.org/abs/2030.12345). 

>![MCL4PHI_figure.jpg](https://github.com/yoheyokubo/Images/blob/096085123e88f741523ad6a0bff298180541e368/MCL4PHI_figure.jpg)

## Requirements
### Codes
Our scripts require a single GPU (but not for PB-LKS). To run the codes, install pytorch_geometric:

```setup
pip install pytorch_geometric
```

Note that if you are goingt to run not only PB-LKS and CL4PHI and MCL4PHI, but also CHERRY, you might want to install pytorch_geometric following [the instruction](https://github.com/pyg-team/pytorch_geometric/discussions/7866#discussioncomment-8829525) so as not to get errors about NeighborSampler.

### Dataset
As for the dataset, we offer several options depending on you interests:
-  If you do not care about the dataset and just want to run our scripts, please download and unzip [data_preprocessed.zip](https://doi.org/10.5281/zenodo.14022091) from Zendo under the *data_cherry* directory.
-  If you are interested in how the preprocessed data was produced, please download and unzip [data_raw.zip](https://doi.org/10.5281/zenodo.14022091) under the same directory and run the sripts in the *code_preprocess* directory. Note that the preprocessing would take several days, maybe a week, to be done.

## Training

To train the model(s) in the paper, run this command for our method (MCL4PHI):

```train
bash cl4phi.sh
```
with *aug="0.0:0.5:0.9:0.99:0.999"* and for CL4PHI with  *aug="0.0"*. PB-LKS and CHERRY can be run in the same way using *pblks.sh* and *cherry.sh*, respectively.

## Evaluation

To evaluate trained models, run:

```eval
bash evaluate.sh
```
Please change the hyperparameter _model_pretrained_ in the file according to which model you want to test.

## Pre-trained Models

Pretrained models (with a seed 123) are included in the *out_cherry* directory.

## Results

Our model achieves the following performance:

>![MCL4PHI_table.png](https://github.com/yoheyokubo/Images/blob/096085123e88f741523ad6a0bff298180541e368/MCL4PHI_table.png) 


## Contributing

If you'd like to contribute, or have any suggestions or questions for this work, you can open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
