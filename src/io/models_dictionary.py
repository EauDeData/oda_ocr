import os

BASE_MODELS = '/data2/users/amolina/oda_ocr_output/ft_from_hiertext/'
FSTRING = "ft{model_name}_from_hiertext/ft{model_name}_from_hiertext.pt"
MODELS_LUT = {
    'Zero-Shot (hiertext)':
       os.path.join('/data2/users/amolina/oda_ocr_output/', 'non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt'),

    'Esposalles':
        os.path.join(BASE_MODELS,
                     FSTRING.format(model_name='esposalles')),
    'iam':
        os.path.join(BASE_MODELS,
                     FSTRING.format(model_name='iam')),
    'gw':
        os.path.join(BASE_MODELS,
                     FSTRING.format(model_name='washington')),
    'Parzival':
        os.path.join(BASE_MODELS,
                     FSTRING.format(model_name='parzival')),
    'CoCoText':
        os.path.join(BASE_MODELS,
                     FSTRING.format(model_name='cocotext')),
    'MLT':
        os.path.join(BASE_MODELS,
                     FSTRING.format(model_name='mlt')),
    # 'TextOCR':
    #     os.path.join(BASE_MODELS,
    #                  FSTRING.format(model_name='textocr')),

}
# BASE_MODELS = '/data2/users/amolina/oda_ocr_output/langs_domain_adaptation/'
# FSTRING = "few_shot_{model_name}_from_averaged_from_hiertext/few_shot_{model_name}_from_averaged_from_hiertext.pt"
#
# MODELS_LUT = {
#     a:
#         os.path.join(BASE_MODELS,
#                      FSTRING.format(model_name=a.lower())) for a in ('Arabic', 'Bangla', 'Chinese', 'Japanese', 'Korean', 'Hindi')
# }
#
#
# _MODELS_LUT = {
#     'Zero-Shot (funsd)':
#         os.path.join(BASE_MODELS, 'scratch_funsd/scratch_funsd.pt'),
#
#
#     'Esposalles (from funsd)':
#         os.path.join(BASE_MODELS,
#                      '{model_name}/{model_name}.pt'.format(model_name='ft_esposalles_from_funsd')),
#     'iam (from funsd)':
#         os.path.join(BASE_MODELS,
#                      '{model_name}/{model_name}.pt'.format(model_name='ft_iam_from_funsd')),
#     'gw (from funsd)':
#         os.path.join(BASE_MODELS,
#                      '{model_name}/{model_name}.pt'.format(model_name='ft_washington_from_funsd')),
#     'Parzival (from funsd)':
#         os.path.join(BASE_MODELS,
#                      '{model_name}/{model_name}.pt'.format(model_name='ft_parzival_from_funsd')),
#     'CoCoText (from funsd)':
#         os.path.join(BASE_MODELS,
#                      '{model_name}/{model_name}.pt'.format(model_name='ft_cocotext_from_funsd')),
#     'MLT (from funsd)':
#         os.path.join(BASE_MODELS,
#                      '{model_name}/{model_name}.pt'.format(model_name='ft_mlt_from_funsd')),
#     'TextOCR (from funsd)':
#         os.path.join(BASE_MODELS,
#                      '{model_name}/{model_name}.pt'.format(model_name='ft_textocr_from_funsd')),
#
# }


'''
MODELS_LUT = {**MODELS_LUT, **{f"Train on HierText -> {lang} (Finetune, non-Taylor)":
                                   os.path.join(BASE_MODELS,
                                                f"non_linear_{lang}_from_hiertext/non_linear_{lang}_from_hiertext.pt")
                               for lang in ['latin', 'arabic', 'japanese', 'bangla', 'chinese', 'korean', 'hindi']
                               }
              }
'''
