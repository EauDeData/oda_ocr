import os

BASE_MODELS = '/data2/users/amolina/oda_ocr_output/'

MODELS_LUT = {
    'Zero-Shot (hiertext)':
        os.path.join(BASE_MODELS, 'non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt'),

    'Esposalles (from hiertext)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ftesposalles_from_hiertext')),
    'iam (from hiertext)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ftiam_from_hiertext')),
    'gw (from hiertext)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ftwashington_from_hiertext')),
    'Parzival (from hiertext)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ftparzival_from_hiertext')),
    # 'CoCoText (from hiertext)':
    #     os.path.join(BASE_MODELS,
    #                  '{model_name}/{model_name}.pt'.format(model_name='ftcocotext_from_hiertext')),
    # 'MLT (from hiertext)':
    #     os.path.join(BASE_MODELS,
    #                  '{model_name}/{model_name}.pt'.format(model_name='ftmlt_from_hiertext')),
    # 'TextOCR (from hiertext)':
    #     os.path.join(BASE_MODELS,
    #                  '{model_name}/{model_name}.pt'.format(model_name='fttextocr_from_hiertext')),

}



_MODELS_LUT = {
    'Zero-Shot (funsd)':
        os.path.join(BASE_MODELS, 'scratch_funsd/scratch_funsd.pt'),


    'Esposalles (from funsd)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ft_esposalles_from_funsd')),
    'iam (from funsd)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ft_iam_from_funsd')),
    'gw (from funsd)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ft_washington_from_funsd')),
    'Parzival (from funsd)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ft_parzival_from_funsd')),
    'CoCoText (from funsd)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ft_cocotext_from_funsd')),
    'MLT (from funsd)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ft_mlt_from_funsd')),
    'TextOCR (from funsd)':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='ft_textocr_from_funsd')),

}


'''
MODELS_LUT = {**MODELS_LUT, **{f"Train on HierText -> {lang} (Finetune, non-Taylor)":
                                   os.path.join(BASE_MODELS,
                                                f"non_linear_{lang}_from_hiertext/non_linear_{lang}_from_hiertext.pt")
                               for lang in ['latin', 'arabic', 'japanese', 'bangla', 'chinese', 'korean', 'hindi']
                               }
              }
'''