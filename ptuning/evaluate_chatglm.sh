MODEL_NAME=../base-model/chatglm-6b
TASK_PATH=benchmark-dataset/test/$1
FILE_NAME=$TASK_PATH/chatglm_test.json

CUDA_VISIBLE_DEVICES=$2 python3 main.py \
    --do_predict \
    --validation_file $FILE_NAME \
    --test_file $FILE_NAME \
    --overwrite_cache \
    --prompt_column prompt \
    --response_column response \
    --model_name_or_path $MODEL_NAME \
    --output_dir $TASK_PATH \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --quantization_bit 4
