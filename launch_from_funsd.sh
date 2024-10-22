#!/bin/bash
echo 'sleeping for 7h in order to wait funsd model to converge'
sleep 7h

# Define dataset keys
datasets=('iam' 'esposalles' 'parzival' 'gw' 'mlt' 'cocotext' 'textocr' 'totaltext' 'washington' 'word_art')

# Define command template

command_template="CUDA_VISIBLE_DEVICES=6 python main.py --use_\$dataset --mlt19_langs Latin --model_architecture vit_atienza --image_height 224 --tokenizer_name oda_giga_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 8 --epoches 30 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name ft_\$dataset_from_funsd --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/scratch_funsd/scratch_funsd.pt"

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
