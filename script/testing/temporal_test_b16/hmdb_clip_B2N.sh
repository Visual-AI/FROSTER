ROOT=PATH_TO_FROSTER_WORKSPACE
CKPT=PATH_TO_FROSTER_WORKSPACE/ckpt/basetraining/B2N_hmdb51_froster
OUT_DIR=$CKPT/testing
LOAD_CKPT_FILE=$ROOT/basetraining/B2N_hmdb51_froster/wa_checkpoints/swa_2_22.pth

# TEST_FILE can be set as val.csv (base set) or test.csv (novel set).
# rephrased_file can be set as train_rephrased.json (base set) or test_rephrased.json (novel set)
# NUM_CLASSES can be set as 26 (base set) or 25 (novel set)
B2N_hmdb_file=B2N_hmdb
TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=val.csv
rephrased_file=train_rephrased.json
NUM_CLASSES=26

cd $ROOT
# DATA.INDEX_LABEL_MAPPING_FILE /root/paddlejob/workspace/env_run/zhouhao/ovclip/FROSTER/label_db/k400-index2cls.json \
# DATA.INDEX_LABEL_MAPPING_FILE /root/paddlejob/workspace/env_run/zhouhao/ovclip/FROSTER/label_rephrase/k400_rephrased_classes.json \

TORCH_DISTRIBUTED_DEBUG=INFO python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_HMDB51.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$B2N_hmdb_file \
    DATA.PATH_PREFIX $ROOT/hmdb51_test \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/B2N_hmdb/$rephrased_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 8 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES $NUM_CLASSES \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 4 \
