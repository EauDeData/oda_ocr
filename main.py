import os
import torch, torchvision
import numpy as np
import wandb
import torch.optim as optim
from vit_pytorch import ViT
import uuid
import collections

from src.io.args import parse_arguments, get_model_name, model_choices_lookup
from src.io.load_datasets import load_datasets
from src.dataloaders.summed_dataloader import CollateFNs
from src.tokenizers.char_tokenizer import CharTokenizer
from src.vision.models import (ViTEncoder, ConvVitEncoder, _ProtoModel, CLIPWrapper, RNNDecoder, ViTAtienzaWrapper,
                               TransformerDecoder)
from src.vision.fusion_models import FusionModelFromEncoderOut
from src.vision.vitstr import vitstr_base_patch16_224
from src.linearize import LinearizedModel, AllMightyWrapper
from src.evaluation.eval import eval_dataset
from src.evaluation.visutils import loop_for_visualization
from src.train_steps.base_ctc import train_ctc, train_ctc_clip, train_reptile
from src.train_steps.base_cross_entropy import train_cross_entropy
torch.manual_seed(42)

def prepare_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Unsupported optimizer choice.")

    return optimizer


def get_lost_and_train(args, tokenizer=None):
    if args.loss_function == 'reptile':
        train_function = train_reptile
        return torch.nn.CTCLoss(blank=tokenizer.tokens[tokenizer.ctc_blank], zero_infinity=True), train_function
    elif args.loss_function == 'ctc':
        train_function = train_ctc_clip
        return torch.nn.CTCLoss(blank=tokenizer.tokens[tokenizer.ctc_blank], zero_infinity=True), train_function
    elif args.loss_function == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(ignore_index=tokenizer.tokens[tokenizer.padding_token]), train_cross_entropy
    elif args.loss_function == 'nll':
        return torch.nn.NLLLoss(), train_cross_entropy

    else:
        raise ValueError('I really wonder how did you reach that point, genuinely.')


def merge_datasets(datasets, split='train'):
    data = datasets[0][split]

    for idx in range(1, len(datasets)):
        if split in datasets[idx]:
            data = data + datasets[idx][split]

    return data


def prepare_tokenizer_and_collator(merged_dataset, transforms, args):
    tokenizer = CharTokenizer(merged_dataset, args.include_eos, args.tokenizer_location,
                              args.tokenizer_name, args.save_tokenizer)
    collator = CollateFNs(args.patch_width, args.image_height, tokenizer,
                          seq2seq_same_size=not args.loss_function == 'ctc', max_size=args.square_image_max_size,
                          transforms=transforms)

    return tokenizer, collator


def prepare_train_loaders(dataset, collator, num_workers, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator.collate,
                                       num_workers=num_workers, shuffle=True)


def prepare_model(vocab_size, args):
    if args.replace_last_layer: vocab_size_arg = args.old_tokenizer_size
    else: vocab_size_arg = vocab_size
    #### LOAD MODEL ###
    feature_size, model = None, None
    url = 'https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_base_patch16_224_aug.pth'

    if args.use_transformers:
        raise NotImplementedError
    else:

        if args.model_architecture == 'vit_encoder_vertical_patch':
            model = ViTEncoder(args.image_height, args.patch_width, 3, args.token_size,
                               [args.visual_tokenizer_width] * args.visual_tokenizer_depth, args.model_depth,
                               args.model_width, vocab_size_arg, args.dropout, args.device)
            feature_size = args.token_size

        elif args.model_architecture == 'conv_vit_encoder':
            assert args.conv_stride == args.patch_width, 'Num tokens will missmatch'
            model = ConvVitEncoder(args.image_height, args.patch_width, 3, args.token_size, args.conv_stride,
                                   args.model_depth, args.model_width, vocab_size_arg, args.dropout, args.device)
            feature_size = args.token_size

        elif args.model_architecture == 'vit_lucid':

            model = _ProtoModel(ViT(
                image_size=args.square_image_max_size,
                patch_size=args.patch_width,
                num_classes=vocab_size_arg,
                dim=args.token_size,
                depth=args.model_depth,
                heads=args.model_width,
                mlp_dim=2048,
                dropout=args.dropout,
                emb_dropout=args.dropout
            ), args.device)
            feature_size = args.token_size
        elif args.model_architecture == 'clip':

            model = CLIPWrapper(vocab_size_arg, args.patch_width, args.device)
            feature_size = 768

        elif args.model_architecture == 'vit_atienza':

            base_model = vitstr_base_patch16_224(pretrained=False)
            base_model_wrapped = ViTAtienzaWrapper(base_model)
            base_model_wrapped.module.vitstr.head = torch.nn.Linear(base_model_wrapped.module.vitstr.head.in_features,
                                                                    96)  # Decretazo, otherwise weights missmatch LoL

            #weights_state_dict = torch.load(model_choices_lookup['atienza_vit_base_augm'])
            #base_model_wrapped.load_state_dict(weights_state_dict)

            if args.decoder_architecture is None:
                base_model_wrapped.module.vitstr.head = torch.nn.Linear(
                                                                    base_model_wrapped.module.vitstr.head.in_features,
                                                                    vocab_size_arg
                )
                base_model_wrapped.module.vitstr.num_classes = vocab_size_arg

            else:
                base_model_wrapped.module.vitstr.head = torch.nn.Linear(
                                                                    base_model_wrapped.module.vitstr.head.in_features,
                                                                    base_model_wrapped.module.vitstr.head.in_features
                )
                feature_size = base_model_wrapped.module.vitstr.head.in_features
                base_model_wrapped.module.vitstr.num_classes = feature_size

            model = _ProtoModel(base_model_wrapped, args.device, target='square_full_images')
            print('Loaded model with:', len(list(model.parameters())), 'modules.')

        elif args.model_architecture == 'encoder_ensemble':

            base_model = vitstr_base_patch16_224(pretrained=url)
            base_model_wrapped = ViTAtienzaWrapper(base_model)
            base_model_wrapped.module.vitstr.head = torch.nn.Linear(base_model_wrapped.module.vitstr.head.in_features,
                                                                    96)  # Decretazo, otherwise weights missmatch LoL

            weights_state_dict = torch.load(model_choices_lookup['atienza_vit_base_augm'])
            base_model_wrapped.load_state_dict(weights_state_dict)
            base_model_wrapped.module.vitstr.head = torch.nn.Linear(
                base_model_wrapped.module.vitstr.head.in_features,
                base_model_wrapped.module.vitstr.head.in_features
            )
            feature_size = base_model_wrapped.module.vitstr.head.in_features
            base_model_wrapped.module.vitstr.num_classes = feature_size
            base_model_wrapper = _ProtoModel(base_model_wrapped, args.device, target='square_full_images')
            base_model_with_decoder = TransformerDecoder(base_model_wrapper, feature_size, args.decoder_token_size,
                                                         args.decoder_depth, vocab_size_arg,
                                       args.decoder_width)

            model = FusionModelFromEncoderOut(base_model_with_decoder, args.checkpoints_list, feature_size,
                                              fusion_width=args.model_width, fusion_depth=args.model_depth,
                                              device=args.device,
                                              seq_size=args.model_fusion_max_tokens,
                                              ocr_dropout_chance=args.ocr_dropout_chance)



    if args.decoder_architecture is not None:
        if args.decoder_architecture == 'transformer':
            model = TransformerDecoder(model, feature_size, args.decoder_token_size, args.decoder_depth, vocab_size_arg,
                                       args.decoder_width)
        else:

            model = RNNDecoder(model, feature_size, args.decoder_token_size, args.decoder_depth, vocab_size_arg,
                               args.decoder_architecture)

    linearized = False
    if args.load_checkpoint and args.checkpoint_name:
        # TODO: Create a --load_linear_checkpoint in order to linearize model BEFORE
        try:
            print(f"Loading state dict from: {args.checkpoint_name}")
            incompatible_keys = model.load_state_dict(torch.load(args.checkpoint_name))
            print(f"(I, script) Found incompatible keys: {incompatible_keys}")
        except:
            print('Load Failed, retrying with linear regime on load')
            print('Linearizing ViT model...')
            wrapped = AllMightyWrapper(non_linear_model=model, device=args.device)
            model = LinearizedModel(wrapped)
            incompatible_keys = model.load_state_dict(torch.load(args.checkpoint_name))
            print(f"(I, script) Found incompatible keys: {incompatible_keys}")
            linearized = True
    if args.replace_last_layer:
        model.lm_head = torch.nn.Linear(args.decoder_token_size, vocab_size)
    ### LINEARIZE ###
    ### The loaded model is already linear?
    if args.linear_model and not linearized:
        print('Linearizing ViT model...')
        wrapped = AllMightyWrapper(non_linear_model=model, device=args.device)
        model = LinearizedModel(wrapped)

    model.to(args.device)
    model.train()
    return model


def evaluation_epoch(datasets, model, tokenizer, collator, args, splits = ['val', 'test', 'train'],
                     eval_fn = eval_dataset):
    evals = []
    for dataset in datasets:
        for split in splits:
            if dataset[split] is not None:
                dataset_name = f"{dataset[split].name}_{dataset[split].split}"
                print(f"Evaluation on {dataset_name} with {len(dataset[split])} samples")
                dataloader = torch.utils.data.DataLoader(dataset[split], batch_size=args.batch_size,
                                                         collate_fn=collator.collate, num_workers=args.num_workers_test)

                evals.append(eval_fn(dataloader, model, dataset_name, tokenizer, wandb))
    return evals


# We will heve to define training strategies for "simple", "continual" and "arithmetic".
## Arithmetic models are trained normally, but they are linearized with the modules

def loop(epoches, model, datasets, collator, tokenizer, args, train_dataloader, optimizer, loss_function,
         train_function=train_ctc, **kwargs):
    ## TEMPORALLY LOOKING BAD. IT SHOULD RECEIVE PROPER ARGS NOT KWARGS ###
    if args.reduce_on_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.reduce_on_plateau,
                                                               threshold=0.01)
    else:
        scheduler = None
    os.makedirs(os.path.join(args.output_folder, args.assigned_uuid), exist_ok=True)
    open(os.path.join(args.output_folder, args.assigned_uuid, 'metadata.txt'), 'w').write(args.model_name_str)

    # collections.OrderedDict is a state dict class

    for epoch in range(epoches):
        print(f"{epoch} / {epoches} epoches")
        if args.loss_function=='reptile' and not isinstance(model, (collections.OrderedDict, dict)):
            otype = type(model)
            model = model.state_dict()
            print(f'Reptile converted model instance type {otype} --> {type(model)}.')
        # loop_for_visualization(train_dataloader, model, tokenizer, None)
        retruned_model = train_function(epoch, train_dataloader, optimizer, model, loss_function, args.patch_width, wandb,
                       tokenizer=tokenizer.tokens[tokenizer.padding_token], scheduler=scheduler,
                       padding_token=tokenizer.tokens[tokenizer.padding_token], max_steps=args.max_train_samples,
                       batch_size=args.batch_size, reptile_args=args)

        if isinstance(model, (collections.OrderedDict, dict)):
            model_template = prepare_model(args.vocabulary_size, args)
            model_template.load_state_dict(model)
        else:
            if not (retruned_model is None):
                model_template = retruned_model
            else:
                model_template = model
        evals = evaluation_epoch(datasets, model_template, tokenizer, collator, args)
        print(evals)

        if not isinstance(model, (collections.OrderedDict, dict)):
            torch.save(model.state_dict(), os.path.join(args.output_folder, args.assigned_uuid, args.assigned_uuid + '.pt'))
        else:
            torch.save(model, os.path.join(args.output_folder, args.assigned_uuid, args.assigned_uuid + '.pt'))

def main(args):
    torch.autograd.set_detect_anomaly(True)
    model_name = get_model_name(args)
    args.model_name_str = model_name
    args.assigned_uuid = str(uuid.uuid4()) if args.output_model_name is None else args.output_model_name
    print(model_name)

    normalize = {
        'normalize': lambda x: (x - x.min()) / max((x.max() - x.min()), 0.01),
        'standarize': lambda x: x / max(x.max(), 1)
    }

    transforms = torchvision.transforms.Compose((
        torchvision.transforms.PILToTensor(),
        normalize['normalize' if not args.standarize else 'standarize'])
    )

    print('Loading all datasets...')
    datasets = load_datasets(args, transforms)
    print(f"Loaded {len(datasets)} datasets")
    if args.loss_function == 'reptile':
        whole_train = [merge_datasets([D], split='train') for D in datasets]
        print(f"Leng of training data: {sum(len(x) for x in whole_train)}" )


    else:
        whole_train = merge_datasets(datasets, split='train')
        print(f"Leng of training data: {len(whole_train)}" )

    if isinstance(whole_train, list):
        tokenizer, collator = prepare_tokenizer_and_collator(whole_train[0], transforms, args)
    else:
        tokenizer, collator = prepare_tokenizer_and_collator(whole_train, transforms, args)

    print('tokenizer num tokens:', len(tokenizer))

    if args.loss_function == 'reptile':
        train_dataloader = [
            prepare_train_loaders(D, collator, args.num_workers_train, args.batch_size) for D in whole_train
        ]
    else:
        train_dataloader = prepare_train_loaders(whole_train, collator, args.num_workers_train, args.batch_size)

    args.vocabulary_size = len(tokenizer)
    args.prepare_model = prepare_model
    model = prepare_model(len(tokenizer), args)
    optimizer = prepare_optimizer(model, args)
    loss_function, train_function = get_lost_and_train(args, tokenizer)

    wandb.init(project='oda_ocr')
    wandb.config.update(args)
    wandb.run.name = model_name

    loop(args.epoches, model, datasets, collator, tokenizer, args, train_dataloader, optimizer, loss_function,
         train_function=train_function)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
