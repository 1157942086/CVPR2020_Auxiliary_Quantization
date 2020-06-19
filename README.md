# Training Quantized Neural Networks with a Full-precision Auxiliary Module

This project hosts the code for implementing the algorithms as presented in our papers:

````
@inproceedings{zhuang2020training,
  title={Training Quantized Neural Networks With a Full-Precision Auxiliary Module},
  author={Zhuang, Bohan and Liu, Lingqiao and Tan, Mingkui and Shen, Chunhua and Reid, Ian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1488--1497},
  year={2020}
}

````
The full paper is available at: https://arxiv.org/abs/1903.11236.

## Training and testing


````
For pretraining, run python pretrain.py

For fine-tuning, run python finetune.py
````

## Pretrained models

For 2-bit ResNet-50 with DoReFa-Net, we achieve 74.0% Top-1 accuracy on ImageNet, where the 32-bit Top-1 accuracy is 76.1%.

The ResNet-50 pretrained model can be downloaded at: https://mega.nz/file/6UphgD5I#TNGA1TSkjjogeQTfxKegqhhxSepccOhITiaLJwyzBVg

More pretrained models will be made available soon.

## Copyright

Copyright (c) Bohan Zhuang. 2020

** This code is for non-commercial purposes only. For commerical purposes,
please contact Bohan Zhuang <bohan.zhuang@monash.edu> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/lice
