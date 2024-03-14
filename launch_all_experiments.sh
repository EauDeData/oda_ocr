#!/bin/bash

# Define dataset keys
datasets=('funsd' 'iam' 'esposalles' 'parzival' 'washington' 'mlt' 'cocotext' 'textocr' 'totaltext' 'washington' 'word_art')

# Define command template

command_template="CUDA_VISIBLE_DEVICES=5 python main.py --use_\$dataset --model_architecture vit_atienza --mlt19_langs Latin --image_height 224 --tokenizer_name oda_giga_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 8 --epoches 60 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name scratch_\$dataset --checkpoint_name /data2/users/amolina/oda_ocr_output/non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt"

# Iterate over datasets
for dataset in "${datasets[@]}"; do
    # Substitute dataset in the command
    current_command="${command_template//\$dataset/$dataset}"
    
    # Print and execute the command
    echo "Running command for dataset: $dataset"
    echo "$current_command"
    echo "CARE; LOAD_CHECKPOINT IS OFF WHEN TRAINING FROM SCRATCH"
    eval "$current_command"
done
