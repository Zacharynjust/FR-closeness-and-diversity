# Emphasizing Closeness and Diversity Simultaneously for Deep Face Representation

## Installation

### 1) Main Requirements

* **pytorch == 1.7.1**
* **torchvision == 0.8.2**
* **CUDA 11**
* **Python 3**

### 2) Clone the repository

```shell
git clone https://github.com/Zacharynjust/FR-closeness-and-diversity.git --recursive
```

### 3)  Install the wheels

```shell
cd FR-closeness-and-diversity
pip install -r requirements.txt
```

## Evaluation

This repo provides a united evaluation script for several FR benchmarks.

Currently, it supports small benchmark verification (e.g., LFW, CFP-FP, and AgeDB) and IJB verification, the code is mainly borrowed from [insightface](https://github.com/deepinsight/insightface),  [MagFace](https://github.com/IrvingMeng/MagFace), and [AdaFace](https://github.com/mk-minchul/AdaFace).  

It also supports MegaFace verification and identification, the code is mainly borrowed from [FFC](https://github.com/tiandunx/FFC). The advantage of this script is that it does not require an [OpenCV-2.4](https://github.com/opencv/opencv/tree/2.4) environment and works much faster; *however, the identification protocol is a little different from the MegaFace official implementation, which may cause a slight performance drop (about 0.05 in our case).*

### 1) Download our pretrained model

[[Baidu Cloud]](https://pan.baidu.com/s/18zOeAKeljeYl8ET_-6xOHg?pwd=jddx)  [[Google Drive]](https://drive.google.com/file/d/12MAany2R5t0RxPWq7T7IaBR-Fz5T_b5L/view?usp=sharing)

### 2) Download evaluation datasets

For small benchmark evaluations (e.g., LFW, CFP-FP, and AgeDB), you can download them from [[Baidu Cloud]](https://pan.baidu.com/s/1pvIOQUB8nYT31-5-45krtQ?pwd=pfod) [[Google Drive]](https://drive.google.com/drive/folders/1RVdrbGbla-OtADhI65KiGYYCz4DxamGC?usp=sharing) and then put `*.bin` in `eval_data` folder.

For IJB evaluation, you can access the datasets via [the official link](https://nigos.nist.gov/datasets/) or [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb). (**Please apply for permissions from [NIST](https://nigos.nist.gov/datasets/) before your usage.**)

For MegaFace evaluation, you can access the datasets and development kit via [the official link](http://megaface.cs.washington.edu/participate/challenge.html) or [[Baidu Cloud]](https://pan.baidu.com/s/10dd26LjhiD_-bUCsJRdgWw?pwd=hw6m). (**Please apply for permissions before your usage.**)

### 3) Customize the config file

Specify the model path and the dataset path in `eval_config.py`

### 4) Run evaluation script

```shell
python evaluate.py --opt eval
```

## Training

<<<<<<< HEAD
### 1) Customize the config file

This repo provides a training script to represent our paper, all you need is to specify training settings in `config.py`. Because a two-stage pipeline is adopted in our paper, we also provide the code to train vanilla ArcFace, see `pretrain.py` 

### 2) Train with ArcFace

Train plain ArcFace for a few epochs (e.g. 4) to get stable difficulty scores. 

Note that you need to manually specify `pretrain` setting entry in `config.py`

```
CUDA_VISIBLE_DEVICES="0,1,2,3" python pretrain.py --opt pretrain
```

### 3) Train with Proposed Framework

In this step, you need to firstly specify model & difficulty score paths in order to resume the pretrained model.

```
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py
```
=======
The training code is coming soon.
>>>>>>> 2945096e1ce76503a240890cde94f4397f125f6e

## Contact

If you have any questions about this work, please contact the author by email at cyzhao@njust.edu.cn

## Cite

If you find this code useful in your research, please consider citing us:

```
@inproceedings{zhao2022emphasizing,
  title={Emphasizing Closeness and Diversity Simultaneously for Deep Face Representation},
  author={Zhao, Chaoyu and Qian, Jianjun and Zhu, Shumin and Xie, Jin and Yang, Jian},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={3430--3446},
  year={2022}
}
```

## Acknowledgment

This repo is built upon several excellent repos, e.g., [insightface](https://github.com/deepinsight/insightface), [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe), [CurricularFace](https://github.com/HuangYG123/CurricularFace), [MagFace](https://github.com/IrvingMeng/MagFace), [AdaFace](https://github.com/mk-minchul/AdaFace), and [FFC](https://github.com/tiandunx/FFC).

