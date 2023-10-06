from src.datasets.ocr.esposalles import EsposalledDataset, DEFAULT_ESPOSALLES

def try_esposalles(base_folder = DEFAULT_ESPOSALLES, split = 'train', cross_val = 'cv1', mode = 'words', image_height = 128, patch_width = 16, transforms = lambda x: x):
    #### ESPOSALLES DATASET TEST ####
    ###### dataset init #######
    dataset = EsposalledDataset(base_folder, split, cross_val, mode, image_height, patch_width, transforms)
    print(dataset[0])