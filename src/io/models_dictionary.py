import os

BASE_MODELS = '/data/users/amolina/oda_ocr_output/'

MODELS_LUT = {
    'Zero-Shot':
        os.path.join(BASE_MODELS, 'non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt'),


    'Esposalles':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='non_linear_esposalles_from_hiertext')),

    'Parzival':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='non_linear_parzival_from_hiertext')),

    'Esposalles -> Parzival':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(model_name='non_linear_parzival__from_esposalles_from_hiertext')),

    'Esposalles + Parzival':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(
                         model_name='non_linear_fused_parzival_esposalles')),
    'Esposalles, Parzival':
        os.path.join(BASE_MODELS,
                     '{model_name}/{model_name}.pt'.format(
                         model_name='non_linear_parzival_and_esposalles_from_hiertext')),
}

'''
MODELS_LUT = {**MODELS_LUT, **{f"Train on HierText -> {lang} (Finetune, non-Taylor)":
                                   os.path.join(BASE_MODELS,
                                                f"non_linear_{lang}_from_hiertext/non_linear_{lang}_from_hiertext.pt")
                               for lang in ['latin', 'arabic', 'japanese', 'bangla', 'chinese', 'korean', 'hindi']
                               }
              }
'''