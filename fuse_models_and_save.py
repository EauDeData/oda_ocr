from src.io.args import parse_arguments, ListToArgsConstructor
from src.task_vectors_original import TaskVector
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
18
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
    # TODO: Do it properly, now I'm just having fun
    # task_vectors = []
    # for weight, model_checkpoint in zip(args.linear_sum_models_weights, args.checkpoints_list):
    #    pass
    # hw_and_scene = '/data2/users/amolina/oda_ocr_output/scene_expert/scene_expert.pt'
    # hw = '/data2/users/amolina/oda_ocr_output/handwritten_expert/handwritten_expert.pt'
    base_model = '/data2/users/amolina/oda_ocr_output/non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt'
    from src.io.models_dictionary import MODELS_LUT

    #hw_and_scene_vector = TaskVector(pretrained_checkpoint=base_model, finetuned_checkpoint=hw_and_scene)
    #hw_only_task_vector = TaskVector(pretrained_checkpoint=base_model, finetuned_checkpoint=hw)
    #negated_hw_task_vector = hw_and_scene_vector + hw_only_task_vector
    #for x in ['Zero-Shot (hiertext)', 'Parzival (from hiertext)', 'MLT (from hiertext)']:
        #del MODELS_LUT[x]# Remove non-orthogonal modules
    print('FUSING MODELS:')
    print(list(MODELS_LUT.keys()))
    vectors = [TaskVector(pretrained_checkpoint=base_model, finetuned_checkpoint=x) for x in MODELS_LUT.values()]
    final_task_vector = sum(vectors[1:], start=vectors[0])
    model = final_task_vector.apply_to(base_model, base_model=model, scaling_coef=(1/len(vectors)))

else: raise ValueError('use --perform_model_arithmetics to fuse models')
args.assigned_uuid = args.output_model_name

os.makedirs(os.path.join(args.output_folder,
                                            args.assigned_uuid),
            exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.output_folder,
                                            args.assigned_uuid,
                                            args.assigned_uuid + '.pt'))