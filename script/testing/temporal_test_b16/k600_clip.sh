ROOT=PATH_TO_FROSTER_WORKSPACE/
CKPT=PATH_TO_FROSTER_WORKSPACE/basetraining/froster_norm
OUT_DIR=$CKPT/testing
LOAD_CKPT_FILE=$ROOT/basetraining/froster_norm/wa_checkpoints/swa_2_22.pth

K600_split=k600_split1
K600_split_class_file=k600_split1_rephrased_classes.json
TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=test.csv
# please modify these two names at the same time
# K600_split can be set as k600_split1 or k600_split2 or k600_split3

cd $ROOT
# MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view'
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$K600_split \
    DATA.PATH_PREFIX  $ROOT/k600_val \
    DATA.PATH_LABEL_SEPARATOR , \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/label_rephrase/$K600_split_class_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 8 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 160 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH /root/.cache/clip/ViT-B-16.pt \
