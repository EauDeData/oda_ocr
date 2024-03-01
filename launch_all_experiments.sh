#!/bin/bash

# Define dataset keys
datasets=('textocr' 'totaltext' 'washington' 'xfund' 'word_art')

# Define command template
command_template="CUDA_VISIBLE_DEVICES=1 python main.py --use_\$dataset --model_architecture vit_atienza --image_height 224 --tokenizer_name oda_giga_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 8 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name ft\$dataset_from_hiertext --load_checkpoint --checkpoint_name /data/users/amolina/oda_ocr_output/non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt"

# Iterate over datasets
for dataset in "${datasets[@]}"; do
    # Substitute dataset in the command
    current_command="${command_template//\$dataset/$dataset}"
    
    # Print and execute the command
    echo "Running command for dataset: $dataset"
    echo "$current_command"
    eval "$current_command"
done
