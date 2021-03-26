# Retina Vessel Segmentation using an Ensemble of U-Nets

![Example Segmentation](/images/Main.png)

## Overview
This provides a software package to automatically segment vasculature from Retinal Fundus photographs. The model uses an ensemble of 10 U-Nets trained on datasets from DRIVE, CHASEdb, and STARE. 

## Installation
This inference code was tested on Ubuntu 18.04.3 LTS, conda version 4.8.0, python 3.7.7, fastai 1.0.61, cuda 10.2, pytorch 1.5.1 and cadene pretrained models 0.7.4. A full list of dependencies is listed in `environment.yml`. 

Inference can be run on the GPU or CPU, and should work with ~4GB of GPU or CPU RAM. For GPU inference, a CUDA 10 capable GPU is required.

For the model weights to download, Github's large file service must be downloaded and installed: https://git-lfs.github.com/ 

This example is best run in a conda environment:

```bash
git lfs clone https://github.com/vineet1992/Retina-Seg/
cd location_of_repo
conda env create -n CXR_Age -f environment.yml
conda activate CXR_Age
python run_model.py dummy_datasets/test_images/ development/models/PLCO_Fine_Tuned_120419 output/output.csv --modelarch=age --type=continuous --size=224
```
Dummy image files are provided in `test_images/;`. Weights for the segmentation model are in `model/UNet_Ensemble_[0-9].pth`. 
Output will be written to `test_output/`.

## Acknowledgements
I thank the NCI and ACRIN for access to trial data, as well as the PLCO and NLST participants for their contribution to research. I would also like to thank the fastai and Pytorch communities as well as the National Academy of Medicine for their support of this work. A GPU used for this research was donated as an unrestricted gift through the Nvidia Corporation Academic Program. The statements contained herein are mine alone and do not represent or imply concurrence or endorsements by the above individuals or organizations.


