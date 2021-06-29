import pdb

from classifiers.ActionClassifier import ClassifierArgs
from classifiers.ThreeLayerMLP import ThreeLayerMLP

def loadClassifier(args):
    cArgs = ClassifierArgs(args)
    classifier = ''
    if cArgs.classifier == '3layerMLP':
        classifier = ThreeLayerMLP(cArgs)
    else:
        print('No classifier created')

    return classifier