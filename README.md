# FROSTER: Frozen CLIP is a Strong Teacher for Open-vocabulary Action Recognition

**This repository is the official implementation of ICLR2024 paper:"FROSTER: Frozen CLIP is a Strong Teacher for Open-vocabulary Action Recognition"**

Xiaohu Huang, Hao Zhou, Kun Yao, Kai Han

 [[`Paper`]](https://openreview.net/pdf?id=zYXFMeHRtO)

# Introduction

In this paper, we introduce FROSTER, an effective framework for open-vocabulary action recognition. The overall pipeline of FROSTER consists of two key components, namely, model finetuning to bridge the gap between image and video tasks, and knowledge distillation to maintain the generalizability of the pretrained CLIP.

<div class="col-sm-12" align=center>
  <img src='figures/method.png' width="90%" height="90%">
</div>

# Performance

We conduct experiments on two open-vocabulary settings, i.e., base-to-novel and cross-dataset. FROSTER achieves state-of-the-art performance on both the two benchmarks.

<div align=center>
 
Table 1: Performance comparison (Top1-Acc (%)) with the CLIP-based methods using ViT-B/16 under the base-to-novel evaluation setting. "HM" denotes the harmonic mean of the accuracy from the base and novel sets. The results of most other papers are taken from Open-VCLIP and ViFi-CLIP. † denotes the results of our implementation. The best results are bolded, and the second best results are $\underline{underlined}$.
| Method | K-400 |  |  | HMDB-51 |  |  | UCF-101 |  |  | SSv2 |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Base | Novel | HM |Base | Novel | HM | Base | Novel | HM | Base | Novel | IMM |
| FrozenCLIP | $62.3$ | $53.4$ | $57.5$ | $53.3$ | $46.8$ | $49.8$ | $78.5$ | $63.6$ | $70.3$ | $4.9$ | $5.3$ | $5.1$ |
| ActionCLIP | $61.0$ | $46.2$ | $52.6$ | $69.1$ | $37.3$ | $48.5$ | $90.1$ | $58.1$ | $70.7$ | $13.3$ | $10.1$ | $11.5$ |
| XCLIP | $74.1$ | $56.4$ | $64.0$ | $69.4$ | $45.5$ | $55.0$ | $89.9$ | $58.9$ | $71.2$ | $8.5$ | $6.6$ | $7.4$ |
| VPT | $69.7$ | $37.6$ | $48.8$ | $46.2$ | $16.0$ | $23.8$ | $90.5$ | $40.4$ | $55.8$ | $8.3$ | $5.3$ | $6.4$ |
| AIM † | $74.6$ | $62.5$ | $68.0$ | $64.0$ | $51.6$ | $57.1$ | $89.8$ | $76.4$ | $82.6$ | $8.5$ | $7.9$ | $8.2$ |
| ST-Adapter † | $73.6$ | $62.0$ | $67.3$ | $65.3$ | $48.9$ | $55.9$ | $85.5$ | $76.8$ | $80.9$ | $9.3$ | $8.4$ | $8.8$ |
| ViFi-CLIP | $76.4$ | $61.1$ | $67.9$ | $\underline{73.8}$ | $\underline{53.3}$ | $\underline{61.9}$ | $92.9$ | $67.7$ | $78.3$ | $\underline{16.2}$ | $\underline{12.1}$ | $\underline{13.9}$ |
| OpenVCLIP | $\underline{76.5}$ | $\underline{62.6}$ | $\underline{68.9}$ | $70.3$ | $50.4$ | $58.7$ | $\underline{94.8}$ | $\underline{77.5}$ | $\underline{85.3}$ | $16.0$ | $11.0$ | ${13.0}$ |
| FROSTER | $\mathbf{77.8}$ | $\mathbf{64.3}$ | $\mathbf{70.4}$ | $\mathbf{74.1}$ | $\mathbf{58.0}$ | $\mathbf{65.1}$ | $\mathbf{95.3}$ | $\mathbf{80.0}$ | $\mathbf{\mathbf{87.0}}$ | $\mathbf{18.3}$ | $\mathbf{12.2}$ | $\mathbf{14.6}$ |
  
</div>

Table 2: Performance comparison (Top1-Acc (%)) with the previous approaches under the cross-dataset evaluation protocol. All methods are based on CLIP ViT-B/16, except for ER-ZASR (TSM pre-trained on ImageNet-1k) and Text4Vis (ViT-L/14). UCF* and HMDB* indicate evaluating the full validation set, while UCF and HMDB denote evaluating across the three validation splits. The results of most other papers are taken from Open-VCLIP and ViFi-CLIP. † denotes the results are produced with our implementation.

<div align="center">
 
| Method | UCF* | UCF | HMDB* | HMDB | K-600 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ER-ZASR | - | $51.8 \pm 2.9$ | - | $35.3 \pm 4.6$ | $42.1 \pm 1.4$ |
| Frozen CLIP † | $74.2$ | $73.8 \pm 0.6$ | $46.3$ | $47.9 \pm 0.5$ | $68.1 \pm 1.1$ |
| ActionCLIP † | $77.4$ | $77.5 \pm 0.8$ | $48.0$ | $48.2 \pm 1.5$ | $62.5 \pm 1.2$ |
| X-CLIP | - | $72.0 \pm 2.3$ | - | $44.6 \pm 5.2$ | $65.2 \pm 0.4$ |
| VPT | - | $69.3 \pm 4.2$ | - | $44.3 \pm 2.2$ | $55.8 \pm 0.7$ |
| Text4Vis| $79.6$ | - | $49.8$ | - | $68.9 \pm 1.0$ |
| AIM †| $79.0$ | $79.4 \pm 1.0$ | $49.5$ | $50.3 \pm 0.8$ | $66.7 \pm 0.5$ |
| ST-Adapter †| $77.9$ | $77.6 \pm 0.7$ | $50.3$ | $51.1 \pm 0.6$ | $60.2 \pm 1.8$ |
| Vita-CLIP | - | $75.0 \pm 0.6$ | - | $48.6 \pm 0.6$ | $67.4 \pm 0.5$ |
| ViFi-CLIP | - | $76.8 \pm 0.7$ | - | $51.3 \pm 0.6$ | $71.2 \pm 1.0$ |
| Open-VCLIP | $\underline{83.5}$ | $\underline{83.4} \pm 1.2$ | $\underline{53.2}$ | $\underline{53.9} \pm 1.2$ | $\underline{73.0} \pm 0.8$ |
| FROSTER | $\mathbf{85.0}$ | $\mathbf{84.8} \pm 1.1$ | $\mathbf{54.5}$ | $\mathbf{54.8} \pm 1.3$ | $\mathbf{74.8} \pm 0.9$ |

</div>


# Dependency

The main dependent packages include: PyTorch 1.11.0 and torchvision 0.12.0 and [`PySlowFast`](https://github.com/facebookresearch/SlowFast)

Detailed Installation instruction can be viewed in [`INSTALL.md`](https://github.com/VisAILab/froster/blob/main/INSTALL.md).

# Data Preparation

- **Kinetics-400.** 

  We obtained the compressed version Kinetics-400 dataset, where videos have been resized to 256, from the [`VoV3d Repo`](https://github.com/youngwanLEE/VoV3D/blob/main/DATA.md#kinetics- 
  400). The repository  provides the download link for the dataset:  [[`Kinetics-400 dataset link`](https://dl.dropbox.com/s/419u0zljf2brsbt/compress.tar.gz)]. After downloading and 
  extracting the data, you should rename the folders "train_256" and "val_256" to "train" and "val" respectively. Additionally, please note that the video "val/crossing_river/ZVdAl- 
  yh9m0.mp4" is invalid and needs to be replaced. You should download a new version of the video from [`here`](https://drive.google.com/file/d/15M07kKQlZEoVzUezppITSnICs83fch8A/view? 
  usp=share_link) and perform the replacement.

- **UCF-101.**

  We download UCF-101 dataset by the [`script`](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101/download_videos.sh) provided by MMAction2.

- **HMDB-51.**

  We donwload HMDB-51 dataset by the [`script`](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hmdb51/download_videos.sh) provided by MMAction2.

- **Kinetics-600 testing.**

  Validation data of Kinetics-600 we used can be donwloaded from [`link`](https://pan.baidu.com/s/1d6wI-n3igMdE1rJ2xP2MsA?pwd=c5mu ).

# Checkpoint

The pre-trained models will be uploaded soon.

# Training

- **Base-to-Novel Setting**
The training scripts are in the **script/training/temporal_b16** folder. 
Please use `train_clip_B2N_hmdb.sh`, `train_clip_B2N_k400.sh`, `train_clip_B2N_ssv2.sh` and `train_clip_B2N_ucf.sh` for the training on HMDB51, K400, SSV2, and UCF101, respectively.

Below is the training script on k400, where you need to modify the `ROOT`, `CKPT`, `DATA.PATH_TO_DATA_DIR`, `DATA.PATH_PREFIX`, `DATA.INDEX_LABEL_MAPPING_FILE`  variables to fit the paths on your server. 

```bash
ROOT=/root/paddlejob/workspace/env_run/output/xiaohu/FROSTER
CKPT=/root/paddlejob/workspace/env_run/output/xiaohu/FROSTER

# TRAIN_FILE can be set as train_1.csv or train_2.csv or train_3.csv;

B2N_k400_file=B2N_k400
TRAIN_FILE=train_1.csv
VAL_FILE=val.csv
TEST_FILE=test.csv

cd $ROOT

TORCH_DISTRIBUTED_DEBUG=INFO python -W ignore -u tools/run_net.py \
  --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_K400.yaml \
  --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$B2N_k400_file \
  TRAIN_FILE $TRAIN_FILE \
  VAL_FILE $VAL_FILE \
  TEST_FILE $TEST_FILE \
  DATA.PATH_PREFIX /root/paddlejob/workspace/env_run/output/xiaohu/data/k400 \
  DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE /root/paddlejob/workspace/env_run/output/xiaohu/FROSTER/zs_label_db/$B2N_k400_file/train_rephrased.json \
  TRAIN.ENABLE True \
  OUTPUT_DIR $CKPT/basetraining/B2N_k400_froster \
  TRAIN.BATCH_SIZE 32 \
  TEST.BATCH_SIZE 240 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  TEST.NUM_SPATIAL_CROPS 1 \
  NUM_GPUS 8 \
  SOLVER.MAX_EPOCH 12 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  SOLVER.BASE_LR 3.33e-6 \
  SOLVER.WARMUP_START_LR 3.33e-8 \
  SOLVER.COSINE_END_LR 3.33e-8 \
  TRAIN.MIXED_PRECISION True \
  DATA.DECODING_BACKEND "pyav" \
  MODEL.NUM_CLASSES 200 \
  MIXUP.ENABLE False \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  MODEL.LOSS_FUNC soft_cross_entropy \
  TRAIN.LINEAR_CONNECT_CLIMB False \
  TRAIN.CLIP_ORI_PATH /root/.cache/clip/ViT-B-16.pt \
  TRAIN.LINEAR_CONNECT_LOSS_RATIO 0.0 \
  MODEL.RAW_MODEL_DISTILLATION True \
  MODEL.KEEP_RAW_MODEL True \
  MODEL.DISTILLATION_RATIO 2.0
```

- **Cross-Dataset Setting**

The training script is also in the **script/training/temporal_b16** folder. 
Please use `train_clip.sh` for the training on K400, where you also need to modify the `ROOT`, `CKPT`, `DATA.PATH_TO_DATA_DIR`, `DATA.PATH_PREFIX`, `DATA.INDEX_LABEL_MAPPING_FILE`  variables to fit the paths on your server. 

# Average the models

To improve the generalizability of your model, after training, you can use `weight_average_tool.py` to average the models from different epochs. The source folder `source_dir` should be changed according to your saved path.

```
python weight_average_tool.py
```

# Evaluation

- **Base-to-Novel Setting**
  
Please use `hmdb_clip_B2N.sh`, `k400_clip_B2N.sh`, `ssv2_clip_B2N.sh` and `ucf_clip_B2N.sh` for the evaluation on HMDB51, K400, SSV2, and UCF101, respectively, where you need to modify the `ROOT`, `CKPT`, `DATA.PATH_TO_DATA_DIR`, `DATA.PATH_PREFIX`, `DATA.INDEX_LABEL_MAPPING_FILE` and `LOAD_CKPT_FILE`  variables to fit the paths on your server.

Below is the evaluation script for k400 dataset.

```bash
ROOT=/root/paddlejob/workspace/env_run/output/xiaohu/FROSTER
CKPT=/root/paddlejob/workspace/env_run/output/xiaohu/FROSTER/basetraining/B2N_k400_froster

OUT_DIR=$CKPT/testing
OAD_CKPT_FILE=/root/paddlejob/workspace/env_run/output/xiaohu/FROSTER/basetraining/B2N_k400_froster/wa_checkpoints/swa_2_22.pth

# TRAIN_FILE can be set as train_1.csv or train_2.csv or train_3.csv;
# TEST_FILE can be set as val.csv (base set) or test.csv (novel set).
# rephrased_file can be set as train_rephrased.json (base set) or test_rephrased.json (novel set)
B2N_k400_file=B2N_k400
TRAIN_FILE=train_1.csv
VAL_FILE=val.csv
TEST_FILE=val.csv
rephrased_file=train_rephrased.json

cd $ROOT

python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_K400.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$B2N_k400_file \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_PREFIX /root/paddlejob/workspace/env_run/output/xiaohu/data/k400 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE /root/paddlejob/workspace/env_run/output/xiaohu/FROSTER/zs_label_db/B2N_k400/$rephrased_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 8 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 200 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 4 \
```

- **Cross-Dataset Setting**

Please use `hmdb_clip.sh`, `ucf_clip.sh`, and `k600_clip.sh` for the evaluation on HMDB51, UCF101, and K600, respectively, where you need to modify the `ROOT`, `CKPT`, `DATA.PATH_TO_DATA_DIR`, `DATA.PATH_PREFIX`, `DATA.INDEX_LABEL_MAPPING_FILE` and `LOAD_CKPT_FILE`  variables to fit the paths on your server.

# Acknowledgement

This repository is built upon [`OpenVCLIP`](https://github.com/wengzejia1/Open-VCLIP), [`PySlowFast`](https://github.com/facebookresearch/SlowFast) and [`CLIP`](https://github.com/openai/CLIP). Thanks for those well-organized codebases.

# Citation

```
@inproceedings{
  huang2024froster,
  title={FROSTER: Frozen CLIP is a Strong Teacher for Open-Vocabulary Action Recognition},
  author={Xiaohu Huang and Hao Zhou and Kun Yao and Kai Han},
  booktitle={International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/pdf?id=zYXFMeHRtO}
}
```
