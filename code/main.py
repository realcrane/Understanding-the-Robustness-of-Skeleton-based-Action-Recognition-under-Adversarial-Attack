# copyright He Wang
# This is the main entrance of the whole project


from datasets.CDataset import *
from classifiers.loadClassifiers import loadClassifier
from Attackers.loadAttackers import loadAttacker
if __name__ == '__main__':

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("-r", "--routine", required=True, help="program to run")
    ap.add_argument("-classifier", "--classifier", required=True, help="choose the classifier to train/test/attack")
    ap.add_argument("-dataset", "--dataset", required=True, help="choose the dataset from hdm05, NTU-60, ...")

    # set up args for ClassifierArgs
    ap.add_argument("-path", "--dataPath", required=True, help="folder to load or save data")
    ap.add_argument("-retPath", "--retPath", required=True, help="folder to load or save data")
    ap.add_argument("-trainFile", "--trainFile", required=False, help="the training data file under --dataPath. "
                    "this is the training data when training the classifier, and the data samples to attack when "
                                                                      "attacking the classifier")
    ap.add_argument("-testFile", "--testFile", required=False, help="the test data file under --dataPath used for "
                                                                    "training the classifier")
    ap.add_argument("-trainedModelFile", "--trainedModelFile", required=False,
                    help="the pre-trained weight file, under --retPath", default='')

    ap.add_argument("-ep", "--epochs", required=False, help="to specify the number of epochs for training",
                    default=200)
    ap.add_argument("-cn", "--classNum", required=True, help="to specify the number of classes")
    ap.add_argument("-bs", "--batchSize", required=False, help="to specify the number of classes",
                    default=32)

    ap.add_argument("-attacker", "--attacker", required=False, help="to specify an adversarail attacker",
                    default='SMART')
    #parse args for SMART attack
    ap.add_argument("-at", "--attackType", required=False, help="to specify the type of attack",
                    default='ab')
    ap.add_argument("-pl", "--perpLoss", required=False, help="to specify the perceptual loss",
                    default='l2')
    ap.add_argument("-ur", "--updateRule", required=False,
                    help="to specify the optimisation method for adversarial attack", default='gd')
    ap.add_argument("-cw", "--classWeight", required=False, help="to specify the weight for classification loss",
                    default=0.6)
    ap.add_argument("-rw", "--reconWeight", required=False, help="to specify the weight for reconstruction loss",
                    default=0.4)
    ap.add_argument("-blw", "--boneLenWeight", required=False, help="to specify the weight for bone length loss",
                    default=0.7)
    # ap.add_argument("-br", "--batchRange", required=False,
    #                 help="to specify the range of batches for adversarial attack", nargs='*', type=int, default=[])
    # ap.add_argument("-bf", "--batchFile", required=False, help="specify pre-trained batch",
    #                 default='')
    # ap.add_argument("-adf", "--attackFile", required=False, help="specify the adversarial example file", default='')
    # ap.add_argument("-tf", "--attackTestFile", required=False, help="load adexample file",
    #                 default='')
    ap.add_argument("-cp", "--clippingThreshold", required=False, help="set up the clipping threshold in update",
                    default=100)

    args = vars(ap.parse_args())
    routine = args["routine"]




    if routine == 'train':
        classifier = loadClassifier(args)
        classifier.train()

    elif routine == 'test':
        classifier = loadClassifier(args)
        classifier.test()

    elif routine == 'gatherCorrectPrediction':
        classifier = loadClassifier(args)
        classifier.collectCorrectPredictions()
    elif routine == 'attack':
        attacker = loadAttacker(args)
        attacker.attack()

    else:
        print('nothing happened')
