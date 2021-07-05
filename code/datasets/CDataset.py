
import torch
from torch.utils.data import Dataset
from shared.helpers import *
from Configuration import *
import numpy as np
class CDataset(Dataset):
    def __init__(self, args, transform=None, target_transform=None):
        data = ''
        if args.routine == 'train' or args.routine == 'attack':
            data = np.load(args.dataFolder + args.trainFile)
        elif args.routine == 'test' or args.routine == 'gatherCorrectPrediction':
            data = np.load(args.dataFolder + args.testFile)
        else:
            print('Unknown routine, cannot create the dataset')
        self.data = torch.from_numpy(data['clips'])
        self.rlabels = data['classes']
        self.labels = torch.from_numpy(self.rlabels).type(torch.int64)
        self.transform = transform
        self.target_transform = target_transform
        self.classNum = args.classNum

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data.to(device), label.to(device)

    # this is the topology of the skeleton, assuming the joints are stored in an array and the indices below
    # indicate their parent node indices. E.g. the parent of the first node is 10 and node[10] is the root node
    # of the skeleton
    parents = np.array([10, 0, 1, 2, 3,
                        10, 5, 6, 7, 8,
                        10, 10, 11, 12, 13,
                        13, 15, 16, 17, 18,
                        13, 20, 21, 22, 23])