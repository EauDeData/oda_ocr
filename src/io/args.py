import argparse
import os
from typing import List, Union, Any

from src.datasets.ocr.test.ocr_datasets_unitary_testing import (DEFAULT_COCOTEXT, DEFAULT_ESPOSALLES, DEFAULT_FUNSD,
                                                                DEFAULT_HIERTEXT, DEFAULT_HIST_MAPS, \
                                                                DEFAULT_IAM, DEFAULT_IIIT, DEFAULT_MLT,
                                                                DEFAULT_PARZIVAL, DEFAULT_SAINT_GALL, \
                                                                DEFAULT_SROIE, DEFAULT_SVT, DEFAULT_TEXTOCR,
                                                                DEFAULT_TOTALTEXT, DEFAULT_WASHINGTON, DEFAULT_XFUND,
                                                                DEFAULT_WA, DEFAULT_AMR,
                                                                DEFAULT_VATICAN, DEFAULT_BORG, DEFAULT_COPIALE)

DEFAULT_OUPUT_FOLDER = '/data2/users/amolina/oda_ocr_output/'

dataset_defaults = {
    'cocotext': DEFAULT_COCOTEXT,
    'esposalles': DEFAULT_ESPOSALLES,
    'funsd': DEFAULT_FUNSD,
    'hiertext': DEFAULT_HIERTEXT,
    'hist_maps': DEFAULT_HIST_MAPS,
    'iam': DEFAULT_IAM,
    'iiit': DEFAULT_IIIT,
    'mlt': DEFAULT_MLT,
    'parzival': DEFAULT_PARZIVAL,
    'saint_gall': DEFAULT_SAINT_GALL,
    'sroie': DEFAULT_SROIE,
    'svt': DEFAULT_SVT,
    'textocr': DEFAULT_TEXTOCR,
    'totaltext': DEFAULT_TOTALTEXT,
    'washington': DEFAULT_WASHINGTON,
    'xfund': DEFAULT_XFUND,
    'word_art': DEFAULT_WA,
    'amr': DEFAULT_AMR,
    'copiale': DEFAULT_COPIALE,
    'borg': DEFAULT_BORG,
    'vatican': DEFAULT_VATICAN
}

model_choices_lookup = {
    'atienza_vit_base_augm': '/data2/users/amolina/oda_ocr_output/models/vitstr_base_patch16_224_aug.pt',
}

model_choices = list(model_choices_lookup.keys()) + [None]

mlt19_lang_choices = ["Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Hindi", "Symbols", "Mixed", "None"]
xfund_lang_choices = ["ZH", "JA", "ES", "FR", "IT", "DE", "PT"]
cocotext_lang_choices = ["english", "not english"]
cocotext_visibility_choices = ['legible', 'illegible']
cv_folds = ['cv1', 'cv2', 'cv3', 'cv4']


class ListToArgsConstructor:
    def __init__(self, listed_datasets, args):
        availability = {dataset_name: dataset_name in dataset_defaults for dataset_name in listed_datasets}
        print(f"Received usage of: {availability}")
        fstring_attrs = ('{}_path', '{}_level', '{}_cross_validation_fold')

        for dataset_name in dataset_defaults:
            setattr(self, f"use_{dataset_name}", dataset_name in listed_datasets)
            for arg_attr in fstring_attrs:
                formatted = arg_attr.format(dataset_name)
                if hasattr(args, formatted): setattr(self, formatted, getattr(args, formatted))

        attributes_to_steal = ('image_height', 'patch_width')

        for attr in attributes_to_steal:
            setattr(self, attr, getattr(args, attr))



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epoches', type=int, default=10)
    parser.add_argument('--reptile_num_episodes', type=int, default=30)
    parser.add_argument('--reptile_tmp_folder', type=str, default='/data2/users/amolina/tmptile/')

    parser.add_argument('--outer_lr', type=float, default=1)

    parser.add_argument('--output_folder', type=str, default=DEFAULT_OUPUT_FOLDER)
    parser.add_argument('--output_model_name', type=str, default=None)
    parser.add_argument('--use_transformers', action='store_true')
    parser.add_argument('--include_eos', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=None)

    parser.add_argument('--perform_model_arithmetics', action='store_true')
    parser.add_argument('--checkpoints_list', nargs='+',
                        help='List of checkpoints to fuse', required=False)
    parser.add_argument('--linear_sum_models_weights', nargs='+',
                        help='Weights of models', type=float, required=False)
    parser.add_argument('--final_vector_scaling', type=float, default=0.8)
    parser.add_argument('--l1penalty', type=float, default=1)

    parser.add_argument('--perform_feature_correction', action='store_true')
    parser.add_argument('--do_democracy', action='store_true')
    parser.add_argument('--do_neuron_inspection', action='store_true')

    parser.add_argument('--model_a_for_dist', default=None)
    parser.add_argument('--model_b_for_dist', default=None)

    parser.add_argument('--metric_a_for_dist', default=None)
    parser.add_argument('--metric_b_for_dist', default=None)

    parser.add_argument('--name_a_for_dist', default=None)
    parser.add_argument('--name_b_for_dist', default=None)

    parser.add_argument('--model_joint_both', default=None)

    parser.add_argument('--source_datasets', nargs='+',
                        help='Source datasets for feature correction', required=False)
    parser.add_argument('--target_datasets', nargs='+',
                        help='Target datasets for feature correction', required=False)
    parser.add_argument('--ocr_dropout_chance', type=float, default=.5)
    parser.add_argument('--datafix_max_tokens', type=int, default=10000)

    dataset_group = parser.add_argument_group('Dataset argument group.')
    parser.add_argument('--model_fusion_max_tokens', type=int, default=-1)

    ### COMMON DATASET ARGS ####
    dataset_group.add_argument('--image_height', type=int, default=128)
    dataset_group.add_argument('--patch_width', type=int, default=16)

    ## DATASET USAGE ##
    # Loop through the dataset_defaults dictionary to add arguments for each dataset
    for dataset_name, default_path in dataset_defaults.items():
        # Create an argument for using the dataset (e.g., --use_cocotext)
        dataset_group.add_argument(f'--use_{dataset_name}', action='store_true')

        # Create an argument for the dataset path (e.g., --cocotext_path)
        dataset_group.add_argument(f'--{dataset_name}_path', type=str, default=default_path)

    ## MULTI LINGUAL DATASET ARGS ##
    dataset_group.add_argument('--mlt19_langs', nargs='*', choices=mlt19_lang_choices, default=mlt19_lang_choices)
    dataset_group.add_argument('--xfund_langs', nargs='*', choices=xfund_lang_choices, default=xfund_lang_choices)
    dataset_group.add_argument('--cocotext_langs', nargs='*', choices=cocotext_lang_choices,
                               default=cocotext_lang_choices)

    ## SOME MODALITY CHOICES ###
    dataset_group.add_argument('--cocotext_visibility', nargs='*', choices=cocotext_visibility_choices,
                               default=['legible'])
    dataset_group.add_argument('--hiertext_handwritten', nargs='*', choices=['true', 'false'],
                               default=['true', 'false'])
    dataset_group.add_argument('--hiertext_include_non_visible', action='store_true')

    # LINES OR WORDS ###
    dataset_group.add_argument('--esposalles_level', choices=['words', 'lines'], default='words')
    dataset_group.add_argument('--hiertext_level', choices=['words', 'lines'], default='words')
    dataset_group.add_argument('--iam_level', choices=['words', 'lines'], default='words')
    dataset_group.add_argument('--washington_level', choices=['word', 'line'], default='word')
    dataset_group.add_argument('--parzival_level', choices=['word', 'line'], default='word')

    # CROSS VALIDATION FOLD ##
    dataset_group.add_argument('--esposalles_cross_validation_fold', choices=cv_folds, default='cv1')
    dataset_group.add_argument('--washington_cross_validation_fold', choices=cv_folds, default='cv1')
    dataset_group.add_argument('--mlt19_cross_validation_fold', choices=cv_folds, default='cv1')
    dataset_group.add_argument('--hist_maps_cross_validation_fold', choices=cv_folds, default='cv1')

    ### TOKENIZER ARGS ###
    dataset_group.add_argument('--tokenizer_name', type=str, default='char_tokenizer')
    dataset_group.add_argument('--tokenizer_location', type=str, default='tmp_/tokenizers/')
    dataset_group.add_argument('--save_tokenizer', action='store_false')

    ### DATALOADER ARGS ###
    dataset_group.add_argument('--num_workers_train', type=int, default=12)
    dataset_group.add_argument('--num_workers_test', type=int, default=6)

    dataset_group.add_argument('--batch_size', type=int, default=224)
    dataset_group.add_argument('--square_image_max_size', type=int, default=224)

    preprocess_group = parser.add_argument_group('Preprocesing argument group.')
    preprocess_group.add_argument('--standarize', action='store_true')

    ### MODEL ARGS ####
    model_group = parser.add_argument_group('Model argument group.')
    model_group.add_argument('--linear_model', action='store_true')
    model_group.add_argument('--replace_last_layer', action='store_true')
    model_group.add_argument('--old_tokenizer_size', type=int, default=5344)

    model_group.add_argument('--model_depth', type=int, default=6)
    model_group.add_argument('--model_width', type=int, default=8)
    model_group.add_argument('--dropout', type=float, default=0.1)
    model_group.add_argument('--token_size', type=int, default=224)

    model_group.add_argument('--visual_tokenizer_depth', type=int, default=0)
    model_group.add_argument('--visual_tokenizer_width', type=int, default=224)

    model_group.add_argument('--load_checkpoint', action='store_true')

    model_group.add_argument('--checkpoint_name', type=str, default=None)

    model_group.add_argument('--model_architecture', type=str,
                             choices=['conv_vit_encoder', 'vit_encoder_vertical_patch', 'vit_lucid', 'clip',
                                      'vit_atienza', 'encoder_ensemble'],
                             default='conv_vit_encoder')
    model_group.add_argument('--conv_stride', type=int, default=8)

    model_group.add_argument('--decoder_architecture', type=str,
                             choices=['lstm', 'rnn', 'gru', None, 'transformer'], default=None)
    model_group.add_argument('--decoder_token_size', type=int, default=224)
    model_group.add_argument('--decoder_depth', type=int, default=1)
    model_group.add_argument('--decoder_width', type=int, default=1)

    ### OPTIM GROUP ####
    optimization_group = parser.add_argument_group('Optimization group')

    optimizer_choices = ['sgd', 'adam', 'adagrad', 'adadelta', 'rmsprop', 'adamw']
    optimization_group.add_argument('--optimizer', choices=optimizer_choices, default='adam', help='Optimizer choice')
    optimization_group.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    # Reptile should not be a loss function...
    optimization_group.add_argument('--loss_function', type=str, default='ctc', choices=['ctc', 'cross_entropy', 'nll',
                                                                                         'reptile'])
    optimization_group.add_argument('--reduce_on_plateau', type=int, default=0)

    return parser.parse_args()


def get_model_name(args):
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize an empty list to store components of the model name
    name_components: list[Union[str, Any]] = []

    # Datasets
    for dataset_name, _ in dataset_defaults.items():
        if getattr(args, f'use_{dataset_name}', False):
            name_components.append(dataset_name)

            # Check if the level is defined for the dataset
            if getattr(args, f'{dataset_name}_level', False):
                name_components.append(getattr(args, f'{dataset_name}_level'))

            # Check if the cross-validation fold is defined for the dataset
            if getattr(args, f'{dataset_name}_cross_validation_fold', False):
                name_components.append(getattr(args, f'{dataset_name}_cross_validation_fold'))

    # Languages
    if args.mlt19_langs and getattr(args, f'use_mlt', False):
        name_components.extend(['mlt_19_langs'] + args.mlt19_langs)
    if args.xfund_langs and getattr(args, f'use_xfund', False):
        name_components.extend(['xfund_langs'] + args.xfund_langs)
    if args.cocotext_langs and getattr(args, f'use_cocotext', False):
        name_components.extend(['cocotext_langs'] + args.cocotext_langs)

    # Other dataset-related options
    if args.hiertext_handwritten and getattr(args, f'use_hiertext', False):
        name_components.append('hiertext_also_handwritten')
    if args.hiertext_include_non_visible and getattr(args, f'use_hiertext', False):
        name_components.append('hiertext_non-visible')
    if args.cocotext_visibility and getattr(args, f'use_cocotext', False):
        name_components.extend(['cocotext_visibility'] + args.cocotext_visibility)

    # Dataloader
    name_components.extend(['batch_size', args.batch_size])

    # Preprocessing
    if args.standarize:
        name_components.append('standarized_images')
    else:
        name_components.append('normalized_images')

    # Model
    if args.linear_model:
        name_components.append('linear_model')
    else:
        name_components.append('non-linear')
    if not args.model_architecture in ['vit_atienza', 'clip']:
        name_components.extend(
            ['depth', args.model_depth, 'width', args.model_width, 'dropout', args.dropout, 'token_size',
             args.token_size])
    name_components.extend(
        ['loss', args.loss_function, 'image_height', args.image_height, 'patch_width', args.patch_width, 'optimizer',
         args.optimizer, 'lr', args.learning_rate])
    if 'conv' in args.model_architecture:
        name_components.extend(['stride', args.conv_stride])
    if args.decoder_architecture is not None:
        name_components.extend(['decoder_architecture', args.decoder_architecture,
                                'decoder_token_size', args.decoder_token_size,
                                'decoder_depth', args.decoder_depth])
        if args.decoder_architecture == 'transformer':
            name_components.extend(['decoder_width', args.decoder_width])
    if args.reduce_on_plateau: name_components.extend(['reduce_on_plateau', args.reduce_on_plateau])

    name_components.extend(['total_square_size', args.square_image_max_size])
    # Join all components to create the model name
    model_name = args.model_architecture + '_' + '_'.join(map(str, name_components))

    return model_name
