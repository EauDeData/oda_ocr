import os

BASE_MODELS = '/data/users/amolina/oda_ocr_output/'

MODELS_LUT = {
    'Train on HierText (zero-shot)':
        os.path.join(BASE_MODELS, 'non_linear_hiertext_only_base/non_linear_hiertext_only_base.pt'),

    'Train on HierText -> All (fused, non-linear, 80%)':
        os.path.join(BASE_MODELS, 'non_linear_fused_hier80p_[all_100p]/non_linear_fused_hier80p_[all_100p].pt'),
}

MODELS_LUT = {**MODELS_LUT, **{f"Train on HierText -> {lang} (Finetune, non-Taylor)":
                                   os.path.join(BASE_MODELS,
                                                f"non_linear_{lang}_from_hiertext/non_linear_{lang}_from_hiertext.pt")
                               for lang in ['latin', 'arabic', 'japanese', 'bangla', 'chinese', 'korean', 'hindi']
                               }
              }
