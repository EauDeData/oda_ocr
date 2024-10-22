#!/bin/bash

langs=("borg" "vatican" "copiale")
model_vals=("esposalles" "iam" "gw" "parzival" "coco" "mlt" "textocr")

for lang in "${langs[@]}"; do
    for T in "${model_vals[@]}"; do
        model_path="/data2/users/amolina/oda_ocr_output/lang_fused_all_models_without_${T}/averaged_model.pth"
        CUDA_VISIBLE_DEVICES=6 python main.py --use_"$lang" --model_architecture vit_atienza --image_height 224 --tokenizer_name oda_giga_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name "without_${T}_finetuned_ft_${lang}" --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name "${model_path}"

    done
done


langs=("Arabic" "Bangla" "Chinese" "Hindi" "Japanese" "Korean" "Symbols")
model_vals=("esposalles" "iam" "gw" "parzival" "coco" "mlt" "textocr")

for lang in "${langs[@]}"; do
    for T in "${model_vals[@]}"; do
        model_path="/data2/users/amolina/oda_ocr_output/lang_fused_all_models_without_${T}/averaged_model.pth"
        CUDA_VISIBLE_DEVICES=6 python main.py --use_mlt --mlt19_langs "$lang" --model_architecture vit_atienza --image_height 224 --tokenizer_name oda_giga_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name  "without_${T}_finetuned_ft_${lang}" --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name "${model_path}"


    done
done
