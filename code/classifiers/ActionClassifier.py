

class ClassifierArgs:
    def __init__(self, args):
        self.dataFolder = args["dataPath"]
        self.retFolder = args["retPath"]
        self.trainFile = args["trainFile"]
        self.testFile = args["testFile"]
        self.trainedModelFile = args["trainedModelFile"]
        self.classNum = int(args["classNum"])
        self.batchSize = int(args["batchSize"])
        self.epochs = int(args["epochs"])
        self.dataset = args["dataset"]
        self.routine = args["routine"]
        self.classifier = args["classifier"]

class ActionClassifier:
    def __init__(self, args):
        self.args = args
        self.loss = ''
        self.model = '';

        # tensorBoardLogger = TensorBoard(log_dir=self.retFolder)
        # self.calbacks = tensorBoardLogger


    def train(self):
        return

    def test(self):
        return

    def collectCorrectPredictions(self):
        return


