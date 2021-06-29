from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.CDataset import *
import numpy as np


def createDataLoader(args):
    trainloader = ''
    validationloader = ''
    testloader = ''
    if args.dataset == 'hdm05':
        if args.routine == 'train':
            traindataset = CDataset(args)

            validation_split = .2
            random_seed = 42
            # Creating data indices for training and validation splits:
            dataset_size = len(traindataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))

            np.random.seed(random_seed)
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            trainloader = DataLoader(traindataset, batch_size=args.batchSize,
                                          sampler=train_sampler)
            validationloader = DataLoader(traindataset, batch_size=args.batchSize,
                                               sampler=valid_sampler)
            if len(args.testFile):
                testdataset = CDataset(args)
                testloader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=False)

        elif args.routine == 'test' or args.routine == 'gatherCorrectPrediction':
            if len(args.testFile):
                testdataset = CDataset(args)
                testloader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=False)

        elif args.routine == 'attack':
            traindataset = CDataset(args)
            trainloader = DataLoader(traindataset, batch_size=args.batchSize, shuffle=False)
    else:
        print ('No dataset is loaded in ThreeLayerMLPArgs')

    return trainloader, validationloader, testloader


