#!/bin/bash

languages=("borg" "copiale" "vatican")
processed_pairs=()

for lang1 in "${languages[@]}"; do
    for lang2 in "${languages[@]}"; do
        if [ "$lang1" != "$lang2" ]; then
            pair1="${lang1}-${lang2}"
            pair2="${lang2}-${lang1}"
            if [[ ! " ${processed_pairs[@]} " =~ " ${pair2} " ]]; then
                cmd="python at_this_point_i_gave_up_refactoring.py --use_$lang1 --use_$lang2"
                cmd+=" --model_a_for_dist '/data2/users/amolina/oda_ocr_output/${lang1,,}_from_ARTH100/${lang1,,}_from_ARTH100.pt'"
                cmd+=" --model_b_for_dist '/data2/users/amolina/oda_ocr_output/${lang2,,}_from_ARTH100/${lang2,,}_from_ARTH100.pt'"
                cmd+=" --metric_a_for_dist 'CER_${lang1}_dataset_test'"
                cmd+=" --metric_b_for_dist 'CER_${lang2}_dataset_test'"
                cmd+=" --name_a_for_dist $lang1 --name_b_for_dist $lang2"
                cmd+=" --model_joint_both '/data2/users/amolina/oda_ocr_output/svd/from_hiertext/averaged_model.pth'"
                cmd+=" --model_architecture vit_atienza --image_height 224 --batch_size 128 --square_image_max_size 224"
                cmd+=" --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001"
                cmd+=" --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name unknown"
                cmd+=" --tokenizer_location /data2/users/amolina/oda_ocr_output/ --tokenizer_name decrypt_tokenizer"
                cmd+=" --load_checkpoint --checkpoint_name '/data2/users/amolina/oda_ocr_output/svd/from_hiertext/averaged_model.pth'  --replace_last_layer --old_tokenizer_size 5344"

                echo "Running command for $lang1 and $lang2"
                eval $cmd

                processed_pairs+=("$pair1")
            fi
        fi
    done
done

