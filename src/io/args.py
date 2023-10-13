import argparse
import os

from src.datasets.ocr.test.ocr_datasets_unitary_testing import (DEFAULT_COCOTEXT, DEFAULT_ESPOSALLES, DEFAULT_FUNSD, DEFAULT_HIERTEXT, DEFAULT_HIST_MAPS,\
                                                                DEFAULT_IAM, DEFAULT_IIIT, DEFAULT_MLT, DEFAULT_PARZIVAL, DEFAULT_SAINT_GALL,\
                                                                DEFAULT_SROIE, DEFAULT_SVT, DEFAULT_TEXTOCR, DEFAULT_TOTALTEXT, DEFAULT_WASHINGTON, DEFAULT_XFUND)

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
    'xfund': DEFAULT_XFUND
}

mlt19_lang_choices = ["Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Hindi", "Symbols", "Mixed", "None"]
xfund_lang_choices = ["ZH", "JA", "ES", "FR", "IT", "DE", "PT"]
cocotext_lang_choices = ["english", "not english"]
cocotext_visibility_choices = ['legible', 'illegible']
cv_folds = ['cv1', 'cv2', 'cv3', 'cv4']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    
    dataset_group = parser.add_argument_group('Dataset argument group.')
    
    ### COMMON DATASET ARGS ####
    dataset_group.add_argument('--image_height', nargs=int, default=128)
    dataset_group.add_argument('--patch_width', nargs=int, default=16)

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
    dataset_group.add_argument('--cocotext_langs', nargs='*', choices=cocotext_lang_choices, default=cocotext_lang_choices)

    ## SOME MODALITY CHOICES ###
    dataset_group.add_argument('--cocotext_visibility', nargs='*', choices=cocotext_visibility_choices, default=['legible'])
    dataset_group.add_argument('--hiertext_handwritten', nargs='*', choices=['true', 'false'], default=['true', 'false'])
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
    dataset_group.add_argument('--tokenizer_name', type = str, default = 'char_tokenizer')
    dataset_group.add_argument('--tokenizer_location', type = str, default = 'tmp_/tokenizers/')
    dataset_group.add_argument('--save_tokenizer', action='store_false')


    ### DATALOADER ARGS ###
    dataset_group.add_argument('--num_workers_train', type = int, default = 12)
    dataset_group.add_argument('--num_workers_test', type = int, default = 6)
    
    dataset_group.add_argument('--batch_size', type = int, default = 224)
    
    preprocess_group =  parser.add_argument_group('Preprocesing argument group.')
    preprocess_group.add_argument('--standarize', action='store_true')
    
    
    ### MODEL ARGS ####
    model_group =  parser.add_argument_group('Model argument group.')
    model_group.add_argument('--linear_model', action='store_true')
    model_group.add_argument('--model_depth', type = int, default=6)
    model_group.add_argument('--model_width', type = int, default=8)
    model_group.add_argument('--dropout', type=float, default = 0.1)
    model_group.add_argument('--token_size', type=int, default=224)
    
    model_group.add_argument('--visual_tokenizer_depth', type=int, default=0)
    model_group.add_argument('--visual_tokenizer_width', type=int, default=224)
    
    model_group.add_argument('--load_checkpoint', action='store_true')
    model_group.add_argument('--checkpoint_name', type=str, default=None)
    
    
    return parser.parse_args()

