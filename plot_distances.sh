#!/bin/bash

languages=("Arabic" "Chinese" "Japanese" "Korean" "Hindi" "Bangla")
processed_pairs=()

for lang1 in "${languages[@]}"; do
    for lang2 in "${languages[@]}"; do
        if [ "$lang1" != "$lang2" ]; then
            pair1="${lang1}-${lang2}"
            pair2="${lang2}-${lang1}"
            if [[ ! " ${processed_pairs[@]} " =~ " ${pair2} " ]]; then
                cmd="python at_this_point_i_gave_up_refactoring.py --use_mlt --mlt19_langs $lang1 $lang2"
                cmd+=" --model_a_for_dist '/data2/users/amolina/oda_ocr_output/langs_domain_adaptation/few_shot_${lang1,,}_from_averaged_from_hiertext/few_shot_${lang1,,}_from_averaged_from_hiertext.pt'"
                cmd+=" --model_b_for_dist '/data2/users/amolina/oda_ocr_output/langs_domain_adaptation/few_shot_${lang2,,}_from_averaged_from_hiertext/few_shot_${lang2,,}_from_averaged_from_hiertext.pt'"
                cmd+=" --metric_a_for_dist 'CER_mlt19_dataset_${lang1:0:3}_val_cv1'"
                cmd+=" --metric_b_for_dist 'CER_mlt19_dataset_${lang2:0:3}_val_cv1'"
                cmd+=" --name_a_for_dist $lang1 --name_b_for_dist $lang2"
                cmd+=" --model_joint_both '/data2/users/amolina/oda_ocr_output/svd/from_hiertext/averaged_model.pth'"
                cmd+=" --model_architecture vit_atienza --image_height 224 --batch_size 128 --square_image_max_size 224"
                cmd+=" --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001"
                cmd+=" --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name unknown"
                cmd+=" --tokenizer_location /data2/users/amolina/oda_ocr_output/ --tokenizer_name oda_giga_tokenizer"
                cmd+=" --load_checkpoint --checkpoint_name '/data2/users/amolina/oda_ocr_output/MULTILINGUAL_from_ARTH100/MULTILINGUAL_from_ARTH100.pt'"

                echo "Running command for $lang1 and $lang2"
                eval $cmd

                processed_pairs+=("$pair1")
            fi
        fi
    done
done

