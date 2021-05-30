# Symmetric Dilated Convolution for Surgical Gesture Recognition (MICCAI 2020)

A PyTorch implementation of the
paper [Symmetric Dilated Convolution for Surgical Gesture Recognition.](https://arxiv.org/pdf/2007.06373.pdf)

### Install

This implementation uses Python 3.6, Pytorch 1.2.0, cudatoolkit 9.2. We recommend to use conda to deploy the environment

Install with conda:

    conda env create -f environment.yml
    conda activate sd-conv

### Dataset

This code is tested on the suturing task of [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
dataset. We use the same data features as [Colin Lea](https://github.com/colincsl/TemporalConvolutionalNetworks). Please
request the Spatial-CNN feature from Colin and put the features (39 extracted video features) into`sd-conv/JIGSAWS/data`
folder.

We also test our method on 50Salads and GTEA datasets, but do not formally benchmark or claim any results.

### Run the code

To train, test, and visulize the result for one split run:

    cd sd-conv
    python train.py

The visualization result is located in `sd-conv/visualization`

### Citation

If you find our work helpful, please consider citing:

    @inproceedings{zhang2020symmetric,
        title={Symmetric Dilated Convolution for Surgical Gesture Recognition},
        author={Zhang, Jinglu and Nie, Yinyu and Lyu, Yao and Li, Hailin and Chang, Jian and Yang, Xiaosong and Zhang, Jian Jun},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={409--418},
        year={2020},
        organization={Springer}
    }
