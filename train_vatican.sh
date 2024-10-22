echo "FINETUNE TIME"

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_FT065 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/both_hw_scene_expert_max_samples_0.65/both_hw_scene_expert_max_samples_5k.pt --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_FT115 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/both_hw_scene_expert_max_samples_1.15/both_hw_scene_expert_max_samples_10k.pt --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_FT1250 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/both_hw_scene_expert_max_samples_12.5/both_hw_scene_expert_max_samples_100k.pt --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_FT6250 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/both_hw_scene_expert_max_samples_62.5/both_hw_scene_expert_max_samples_500k.pt --replace_last_layer --old_tokenizer_size 5344

echo "now for the fused models"

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_ARITH1065 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/0.65_fusion/averaged_model.pth --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_ARITH115 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/1.15_fusion/averaged_model.pth --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_ARITH11250 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/12.5_fusion/averaged_model.pth --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_ARITH16250 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/subsampling/62.5_fusion/averaged_model.pth --replace_last_layer --old_tokenizer_size 5344
# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_BASELINE --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_FT100 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/both_hw_scene_expert_from_hiertext/both_hw_scene_expert_from_hiertext.pt --replace_last_layer --old_tokenizer_size 5344

# CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name vatican_from_ARTH100 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/svd/from_hiertext/averaged_model.pth --replace_last_layer --old_tokenizer_size 5344

CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --use_borg --use_copiale --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name ALL_CIPHERS_from_ARTH100 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/svd/from_hiertext/averaged_model.pth --replace_last_layer --old_tokenizer_size 5344

CUDA_VISIBLE_DEVICES=5 python  main.py --use_vatican  --use_borg --use_copiale  --model_architecture vit_atienza --image_height 224 --tokenizer_name decrypt_tokenizer --batch_size 128 --square_image_max_size 224 --loss_function ctc --num_workers_train 4 --num_workers_test 4 --epoches 30 --learning_rate 0.0001 --reduce_on_plateau 5 --decoder_architecture transformer --output_model_name ALL_CIPHERS_from_FT100 --tokenizer_location /data2/users/amolina/oda_ocr_output/ --load_checkpoint --checkpoint_name /data2/users/amolina/oda_ocr_output/both_hw_scene_expert_from_hiertext/both_hw_scene_expert_from_hiertext.pt --replace_last_layer --old_tokenizer_size 5344
