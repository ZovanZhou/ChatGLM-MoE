GPU_ID=$1
TARGET_TASK=$4
CHECKPOINT_PATH=./benchmark-dataset/checkpoints/$TARGET_TASK

PRE_SEQ_LEN=128
PRE_N_EXPERTS=2
LR=1e-2

BASE_PATH=./benchmark-dataset/train

TRAIN_FILE=$BASE_PATH/$2/chatglm_train_data.json
PRE_CUR_EXPERT=1
CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py \
    --do_train \
    --train_file $TRAIN_FILE \
    --validation_file $TRAIN_FILE \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path ../base-model/chatglm-6b \
    --output_dir $CHECKPOINT_PATH \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --prefix_n_experts $PRE_N_EXPERTS \
    --prefix_cur_expert $PRE_CUR_EXPERT \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \

PTUNING_CHECKPOINT=$CHECKPOINT_PATH/checkpoint-3000
TRAIN_FILE=$BASE_PATH/$3/chatglm_train_data.json
PRE_CUR_EXPERT=2
CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py \
    --do_train \
    --train_file $TRAIN_FILE \
    --validation_file $TRAIN_FILE \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path ../base-model/chatglm-6b \
    --ptuning_checkpoint $PTUNING_CHECKPOINT \
    --output_dir $CHECKPOINT_PATH \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --prefix_n_experts $PRE_N_EXPERTS \
    --prefix_cur_expert $PRE_CUR_EXPERT \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \