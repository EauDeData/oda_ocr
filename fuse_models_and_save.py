from src.io.args import parse_arguments, ListToArgsConstructor
from evaluation_protocol import fuse_models
from main import prepare_tokenizer_and_collator, prepare_model

import torchvision
import torch
import os

'''
Example command for lazy ahh ppl:

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python fuse_models_and_save.py \
    --load_checkpoint --checkpoint_name "/data/users/amolina/oda_ocr_output/non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt" \
    --checkpoints_list '/data/users/amolina/oda_ocr_output/korean_from_hiertext_nonlinear/korean_from_hiertext_nonlinear.pt'\
    '/data/users/amolina/oda_ocr_output/korean_from_hiertext_nonlinear/korean_from_hiertext_nonlinear.pt' --tokenizer_name oda_giga_tokenizer\
    --model_architecture vit_atienza --decoder_architecture transformer --output_model_name 'non_linear_fused[hier_base_50p_kor25p_arab25p]'\
    --linear_sum_models_weights 0.5 0.5 --final_vector_scaling 0.5 --perform_model_arithmetics


'''

normalize = {
    'normalize': lambda x: (x - x.min()) / max((x.max() - x.min()), 0.01),
    'standarize': lambda x: x / max(x.max(), 1)
}

args = parse_arguments()

transforms = torchvision.transforms.Compose((
    torchvision.transforms.PILToTensor(),
    normalize['normalize' if not args.standarize else 'standarize'])
)

tokenizer, collator = prepare_tokenizer_and_collator(None, transforms, args)
model = prepare_model(len(tokenizer), args) # Here we define the base model

if args.perform_model_arithmetics:
    model_vectorized = fuse_models(model, len(tokenizer), args)
else: raise ValueError('use --perform_model_arithmetics to fuse models')
args.assigned_uuid = args.output_model_name

os.makedirs(os.path.join(args.output_folder,
                                            args.assigned_uuid),
            exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.output_folder,
                                            args.assigned_uuid,
                                            args.assigned_uuid + '.pt'))