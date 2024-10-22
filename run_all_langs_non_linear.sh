#!/bin/bash

langs=("Arabic" "Bangla" "Chinese" "Hindi" "Japanese" "Korean" "Symbols")
T_vals=("1" "3" "5" "15" "25" "75")
K_vals=("1" "3" "5" "15" "25" "75")

for lang in "${langs[@]}"; do
    for T in "${T_vals[@]}"; do
        for K in "${K_vals[@]}"; do
            if [ $((T * K)) -eq 75 ]; then
                model_path="/data2/users/amolina/oda_ocr_output/reptile_both_1_outer_step_episode_len_${K}_epoches_${T}/reptile_both_1_outer_step_episode_len_${K}_epoches_${T}.pt"
                if [ ! -f "$model_path" ]; then
                    echo "This model should be trained: ${K}_${T}_reptile_finetuned_ft_${lang}"
                else
                    echo "Running command for T=$T, K=$K, and language $lang"
                CUDA_VISIBLE_DEVICES=6 python main.py --use_mlt --mlt19_langs "$lang" --model_architecture vit_atienza --image_height 224 --tokenizer_name oda_giga_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name "${K}_${T}_reptile_finetuned_ft_${lang}" --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name "/data2/users/amolina/oda_ocr_output/reptile_both_1_outer_step_episode_len_${K}_epoches_${T}/reptile_both_1_outer_step_episode_len_${K}_epoches_${T}.pt"
                fi
            fi
        done
    done
done
