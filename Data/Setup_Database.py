import os
from Data.PreProcessV2 import preprocess
from torch.utils.data import DataLoader
from Data.Dataset import data


def setup_database(isTrain, normalize, batchSize, dataRoot='../101_ObjectCategories'):
    file = 'train_dataColor.data' if isTrain else 'val_dataColor.data'
    if not os.path.isfile(os.path.join(dataRoot, file)):
        print("Pre-processing dataset")
        # Preprocess & normalise data
        preprocess(dataRoot)

    # make dataset

    print('making dataset')
    dataset = data(dataRoot, isTrain)
    print('making dataloader')
    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True,
                            num_workers=3)  # --> num-workers = 2 in queue before processing

    return dataLoader
