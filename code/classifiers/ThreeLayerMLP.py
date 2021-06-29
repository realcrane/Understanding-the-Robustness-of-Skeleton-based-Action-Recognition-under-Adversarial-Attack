
from classifiers.ActionClassifier import ActionClassifier
import numpy as np
import os
import torch
from torch import nn

from shared.helpers import *
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloaders import *

import pdb
class ThreeLayerMLP(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.validationloader, self.testloader = createDataLoader(args)
        self.createModel()
        self.configureOptimiser()
        self.classificationLoss()
    def createModel(self):
        class Classifier(nn.Module):
            def __init__(self, dataloader):
                super().__init__()

                self.dataShape = dataloader.dataset.data.shape
                self.flatten = nn.Flatten()
                self.mlpstack = nn.Sequential(
                    nn.Linear(self.dataShape[1] * self.dataShape[2], 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, dataloader.dataset.classNum),
                    nn.ReLU()
                )
            def forward(self, x):
                x = self.flatten(x)
                logits = self.mlpstack(x)
                return logits

        # create the traindata loader, if the routine is 'attack', then the data will be attacked in an Attacker
        if self.args.routine == 'train' or self.args.routine == 'attack':
            self.model = Classifier(self.trainloader)
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction':
            self.model = Classifier(self.testloader)
        if len(self.args.trainedModelFile) > 0:
            self.model.load_state_dict(torch.load(self.args.retFolder + self.args.trainedModelFile))

    def configureOptimiser(self):

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)

    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    #this function is to train the classifier from scratch
    def train(self):
        size = len(self.trainloader.dataset)

        bestLoss = np.infty
        bestValLoss = np.infty

        logger = SummaryWriter()

        for ep in range(self.args.epochs):
            epLoss = 0
            batchNum = 0
            for batch, (X, y) in enumerate(self.trainloader):
                batchNum += 1
                # Compute prediction and loss
                pred = self.model(X)
                loss = self.classLoss(pred, y)

                epLoss += loss
                # Backpropagation
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if batch % 50 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {ep}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

            # save a model if the best training loss so far has been achieved.
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            if epLoss < bestLoss:
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                torch.save(self.model.state_dict(), self.args.retFolder+'minLossModel.pth')
                bestLoss = epLoss

            # run validation and save a model if the best validation loss so far has been achieved.
            valLoss = 0
            vbatch = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.validationloader):
                pred = self.model(tx)
                valLoss += self.classLoss(pred, ty)
                vbatch += 1

            valLoss /= vbatch
            logger.add_scalar('Loss/validation', valLoss, ep)
            self.model.train()
            if valLoss < bestValLoss:
                print(f"epoch: {ep} per epoch average validation loss improves from: {bestValLoss} to {valLoss}")
                torch.save(self.model.state_dict(), self.args.retFolder+'minValLossModel.pth')
                bestValLoss = valLoss

    #this function tests the trained classifier and also save correctly classified samples in 'adClassTrain.npz' for
    #further adversarial attack
    def test(self):
        if len(self.args.trainedModelFile) == 0 or self.testloader == '':
            print('no pre-trained model to load')
            return

        self.model.eval()

        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            results[v*self.args.batchSize:(v+1)*self.args.batchSize] = pred
            diff = (pred - ty) != 0
            misclassified += torch.sum(diff)

        error = misclassified / len(self.testloader.dataset)
        print(f"accuracy: {1-error:>4f}")
        np.savetxt(self.args.retFolder + 'testRets.txt', results)
        np.savetxt(self.args.retFolder + 'testGroundTruth.txt', self.testloader.dataset.rlabels)

        # scores = self.model.evaluate(inputData, labels, batch_size=self.batch_size)

        # print('%s: %.2f' % (self.model.metrics_names[0], scores))

        # hitIndices = []
        #
        # for i in range(0, len(results)):
        #     if rlabels[i] == results[i]:
        #         hitIndices.append(i)
        #
        # hitIndices = np.array(hitIndices)
        #
        # adData = inputData[hitIndices]
        # adLabels = rlabels[hitIndices]
        #
        # print('Accurary: %.2f' % (len(hitIndices) / len(results)))
        # # validBatchNo = 3
        # # validSplit = int(len(adData) / self.batch_size - validBatchNo) * self.batch_size
        #
        # # trainData = adData[0: validSplit]
        # # trainLabels = adLabels[0: validSplit]
        #
        # # validData = adData[validSplit:validSplit+(validBatchNo*self.batch_size)]
        # # validLabels = adLabels[validSplit:validSplit+(validBatchNo*self.batch_size)]
        #
        # np.savez_compressed(self.args.dataFolder + 'adClassTrain.npz', clips=adData, classes=adLabels)
        # np.savez_compressed(self.dataFolder + 'adClassTest.npz', clips=validData, classes=validLabels)

    # this function is to collected all the testing samples that can be correctly collected
    # by the pre-trained classifier, to make a dataset for adversarial attack
    def collectCorrectPredictions(self):
        if len(self.args.trainedModelFile) == 0 or self.testloader == '':
            print('no pre-trained model to load')
            return

        self.model.eval()

        # collect data from the training data
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff

        adData = self.testloader.dataset.data[results.astype(bool)]
        adLabels = self.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")

        path = self.args.retFolder + self.args.classifier
        if not os.path.exists(path):
            os.mkdir(path)
        np.savez_compressed(path+'/adClassTrain.npz', clips=adData, classes=adLabels)

    def testAAFile(self, modelFile='', testFile=''):

        data = np.load(self.dataFolder + testFile)

        motions = data['clips']
        # orMotions = data['oriClips']
        # plabels = data['classes']
        tlabels = data['tclasses']

        self.dataShape = motions.shape

        self.createModel()

        if len(modelFile) > 0:
            self.model.load_weights(self.retFolder + modelFile, by_name=True)
        else:
            print('No model loaded.')
            return

        self.configureOptimiser()

        self.classificationLoss()

        self.model.compile(loss=self.classLoss, optimizer=self.optimiser)

        start = 0
        end = int(len(motions) / self.batch_size)

        sum = 0
        for batchNo in range(start, end):

            print('batchNo %d/%d' % (batchNo, end))

            rresults = self.model.predict(motions[batchNo * self.batch_size:(batchNo + 1) * self.batch_size],
                                          batch_size=self.batch_size)

            results = np.argmax(rresults, axis=-1)

            hitIndices = []

            for i in range(0, len(results)):
                if tlabels[i] != results[i]:
                    hitIndices.append(i)

            hitIndices = np.array(hitIndices)

            sum += len(hitIndices) / len(results)

        print('Accurary: %.4f' % (sum / (end - start)))

    def adTest(self, modelFile='', adExampleFile=''):

        subFolders = [f for f in list(os.listdir(self.dataFolder)) if re.search('^batch', f) != None]

        if len(adExampleFile) == 0:

            for rfolder in subFolders:

                maxsa = 0
                maxus = 0
                maxab = 0

                elements = rfolder.split('_')

                at = elements[1]

                files = [f for f in list(os.listdir(self.dataFolder + '/' + rfolder))]

                for file in files:
                    felements = os.path.splitext(file)[0].split('_')

                    sr = felements[-1]
                    if sr == 'rets':
                        continue

                    if at == 'ab':
                        if float(sr) > maxab:
                            maxab = float(sr)
                            adExampleFile = file
                    elif at == 'us':
                        if float(sr) > maxus:
                            maxus = float(sr)
                            adExampleFile = file
                    elif at == 'sa':
                        if float(sr) > maxsa:
                            maxsa = float(sr)
                            adExampleFile = file
                    else:
                        print('Unknown attack type')

                print('Processing %s' % (self.dataFolder + '/' + rfolder + '/' + adExampleFile))
                data = np.load(self.dataFolder + '/' + rfolder + '/' + adExampleFile)

                inputData = data['clips']
                rlabels = data['classes']

                labels = to_categorical(rlabels, self.classNum)

                inputData = inputData[0: int(len(inputData) / self.batch_size) * self.batch_size]
                labels = labels[0: int(len(labels) / self.batch_size) * self.batch_size]

                self.dataShape = inputData.shape

                if self.model == '':
                    self.createModel()

                    if len(modelFile) > 0:
                        self.model.load_weights(self.retFolder + modelFile, by_name=True)
                    else:
                        print('No model loaded.')
                        return

                    self.configureOptimiser()

                    self.classificationLoss()

                    self.model.compile(loss=self.classLoss, optimizer=self.optimiser)

                rresults = self.model.predict(inputData, batch_size=self.batch_size)

                np.savez_compressed(self.dataFolder + '/' + rfolder + '/' + adExampleFile + '_rets.npz',
                                    rawRets=rresults, tlabels=data['tclasses'])

        else:

            data = np.load(self.dataFolder + adExampleFile)

            inputData = data['clips']
            rlabels = data['classes']

            labels = to_categorical(rlabels, self.classNum)

            inputData = inputData[0: int(len(inputData) / self.batch_size) * self.batch_size]
            labels = labels[0: int(len(labels) / self.batch_size) * self.batch_size]

            self.dataShape = inputData.shape

            self.createModel()

            if len(modelFile) > 0:
                self.model.load_weights(self.retFolder + modelFile, by_name=True)
            else:
                print('No model loaded.')
                return

            self.configureOptimiser()

            self.classificationLoss()

            self.model.compile(loss=self.classLoss, optimizer=self.optimiser)

            rresults = self.model.predict(inputData, batch_size=self.batch_size)

            np.savez_compressed(self.dataFolder + adExampleFile + '_rets.npz', rawRets=rresults,
                                tlabels=data['tclasses'])

    def createDiscriminator(self, dataShape, batch_size):
        shape = dataShape
        totalInput = Input(batch_shape=(batch_size, shape[0], shape[1], shape[2]))

        conv1 = Conv2D(32, 3, activation='sigmoid', data_format='channels_first')(totalInput)
        conv2 = Conv2D(32, 3, activation='sigmoid', data_format='channels_first')(conv1)
        fcn = Dense(1, activation='sigmoid')(Flatten()(conv2))

        model = Model(totalInput, fcn)

        self.discriminator = model
        self.discriminator.compile(loss=self.adversarialLoss, optimizer=self.optimiser)

    def updateFeatureMap(self, data):
        parents = [[10], [0], [1], [2], [3],
                   [10], [5], [6], [7], [8],
                   [10], [10], [11], [12], [13],
                   [13], [15], [16], [17], [18],
                   [13], [20], [21], [22], [23]]

        positions = tf.reshape(data, (data.shape[0], data.shape[1], -1, 3))

        positions = tf.transpose(positions, [2, 1, 0, 3])

        boneParents = tf.gather_nd(
            positions,
            parents,
            name=None,
            batch_dims=0
        )

        randIndices = np.random.permutation(self.batch_size)

        boneVecs = positions - boneParents

        boneVecs = tf.transpose(boneVecs, [2, 1, 0, 3])

        for i in range(self.batch_size):
            j = randIndices[i]
            cosDist = tf.tensordot(boneVecs[i, j], tf.transpose(boneVecs[i, j]), axes=1)
            norm1 = tf.math.sqrt(K.sum(K.square(boneVecs[i, j]), axis=-1))
            norm2 = tf.tensordot(norm1, tf.transpose(norm1), axes=0)

            self.frameFeature[i, 0, :, :].assign(cosDist / norm2)




