import pdb
import os
import torch
from Attackers.Attacker import ActionAttacker
from classifiers.loadClassifiers import loadClassifier
import torch as K
import numpy as np
from shared.helpers import MyAdam, to_categorical

class SmartAttacker(ActionAttacker):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.name = 'SMART'
        #parameters for SMART attack
        self.perpLossType = args["perpLoss"]
        self.classWeight = float(args["classWeight"])
        self.reconWeight = float(args["reconWeight"])
        self.boneLenWeight = float(args["boneLenWeight"])
        # self.batchRange = args["batchRange"]
        # self.batchFile = args["batchFile"]
        self.attackType = args["attackType"]
        self.epochs = int(args["epochs"])
        self.updateRule = args["updateRule"]
        self.updateClip = float(args["clippingThreshold"])
        self.deltaT = 1 / 30
        self.topN = 3
        self.refBoneLengths = []
        self.optimizer = ''
        self.classifier = loadClassifier(args)

    def boneLengthLoss (self, parentIds, adData, refBoneLengths):

        # parents = [[10], [0], [1], [2], [3],
        #            [10], [5], [6], [7], [8],
        #            [10], [10], [11], [12], [13],
        #            [13], [15], [16], [17], [18],
        #            [13], [20], [21], [22], [23]]

        # convert the data into shape (batchid, frameNo, jointNo, jointCoordinates)
        jpositions = K.reshape(adData, (adData.shape[0], adData.shape[1], -1, 3))


        boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :] + 1e-8

        boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

        boneLengthsLoss = K.mean(
            K.sum(K.sum(K.square(boneLengths - refBoneLengths), axis=-1), axis=-1))
        return boneLengthsLoss
    # def velLoss (self, adData, refData):
    #     refVel = refData[:, 1:, :] - refData[:, :-1, :] / self.deltaT
    #
    #     adVel = adData[:, 1:, :] - adData[:, :-1, :] / self.deltaT
    #
    #     velLoss = K.mean(K.sum(K.sum(K.square(adVel - refVel), axis=-1), axis=-1), axis=-1)
    #
    #     return velLoss
    def accLoss (self, adData, refData, jointWeights = None):
        refAcc = (refData[:, 2:, :] - 2 * refData[:, 1:-1, :] + refData[:, :-2, :]) / self.deltaT / self.deltaT

        adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / self.deltaT / self.deltaT

        if jointWeights == None:
            return K.mean(K.sum(K.sum(K.square(adAcc - refAcc), axis=-1), axis=-1), axis=-1)
        else:
            return K.mean(K.sum(K.sum(K.square(adAcc - refAcc), axis=-1), axis=-1)*jointWeights, axis=-1)
    # def jerkLoss (self, adData, refData):
    #     refJerk = (refData[:, 3:, :] - 3 * refData[:, 2:-1, :] + 3 * refData[:, 1:-2, :] + refData[:, :-3,
    #                                                                                    :]) / self.deltaT / self.deltaT / self.deltaT
    #     adJerk = (adData[:, 3:, :] - 3 * adData[:, 2:-1, :] + 3 * adData[:, 1:-2, :] + adData[:, :-3,
    #                                                                                    :]) / self.deltaT / self.deltaT / self.deltaT
    #
    #     jerkLoss = K.mean(K.sum(K.sum(K.square(adJerk - refJerk), axis=-1), axis=-1), axis=-1)
    #
    #     return jerkLoss

    def perceptualLoss(self, refData, adData, refBoneLengths):

        # the joint weights are decided per joint, the spinal joints have higher weights.
        jointWeights = torch.Tensor([[[0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.04, 0.04, 0.04, 0.04, 0.04,
                                      0.02, 0.02, 0.02, 0.02, 0.02,
                                      0.02, 0.02, 0.02, 0.02, 0.02]]])
        # parents = [[10], [0], [1], [2], [3],
        #            [10], [5], [6], [7], [8],
        #            [10], [10], [11], [12], [13],
        #            [13], [15], [16], [17], [18],
        #            [13], [20], [21], [22], [23]]

        elements = self.perpLossType.split('_')

        if elements[0] == 'l2' or elements[0] == 'l2Clip':

            diffmx = K.square(refData - adData),
            squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                                axis=-1)

            weightedSquaredLoss = squaredLoss * jointWeights

            squareCost = K.sum(K.sum(weightedSquaredLoss, axis=-1), axis=-1)

            oloss = K.mean(squareCost, axis=-1)



        elif elements[0] == 'lInf':
            squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                                axis=-1)

            weightedSquaredLoss = squaredLoss * jointWeights

            squareCost = K.sum(weightedSquaredLoss, axis=-1)

            oloss = K.mean(K.norm(squareCost, ord=np.inf, axis=0))

        else:
            print('warning: no reconstruction loss')
            return

        if len(elements) == 1:
            return oloss

        # elif elements[1] == 'acc':
        #     jaccLoss = self.accLoss(adData, refData)
        #
        #     return jaccLoss * (1 - self.reconWeight) + oloss * self.reconWeight
        #
        # elif elements[1] == 'smoothness':
        #     adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / self.deltaT / self.deltaT
        #
        #     jointAcc = K.mean(K.sum(K.sum(K.square(adAcc), axis=-1), axis=-1), axis=-1)
        #
        #     return jointAcc * (1 - self.reconWeight) + oloss * self.reconWeight
        #
        # elif elements[1] == 'smoothness-bone':
        #     adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / self.deltaT / self.deltaT
        #
        #     squaredLoss = K.sum(K.reshape(K.square(adAcc), (adAcc.shape[0], adAcc.shape[1], 25, -1)), axis=-1)
        #
        #     weightedSquaredLoss = squaredLoss * jointWeights
        #
        #     jointAcc = K.mean(K.sum(K.sum(weightedSquaredLoss, axis=-1), axis=-1), axis=-1)
        #
        #     boneLengthsLoss = self.boneLengthLoss(parents, adData, refBoneLengths)
        #
        #     return boneLengthsLoss * (1 - self.reconWeight) * self.boneLenWeight + jointAcc * (1 - self.reconWeight) * (
        #                 1 - self.boneLenWeight) + oloss * self.reconWeight
        #
        #
        # elif elements[1] == 'jerkness':
        #
        #     adJerk = (adData[:, 3:, :] - 3 * adData[:, 2:-1, :] + 3 * adData[:, 1:-2, :] + adData[:, :-3,
        #                                                                                    :]) / self.deltaT / self.deltaT / self.deltaT
        #
        #     jointJerk = K.mean(K.sum(K.sum(K.square(adJerk), axis=-1), axis=-1), axis=-1)
        #
        #     return jointJerk * (1 - self.reconWeight) + oloss * self.reconWeight
        #
        # elif elements[1] == 'acc-jerk':
        #
        #     refAcc = (refData[:, 2:, :] - 2 * refData[:, 1:-1, :] + refData[:, :-2, :]) / self.deltaT / self.deltaT
        #
        #     adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / self.deltaT / self.deltaT
        #
        #     jointAcc = K.mean(K.sum(K.sum(K.square(adAcc - refAcc), axis=-1), axis=-1), axis=-1)
        #
        #     adJerk = (adData[:, 3:, :] - 3 * adData[:, 2:-1, :] + 3 * adData[:, 1:-2, :] + adData[:, :-3,
        #                                                                                    :]) / self.deltaT / self.deltaT / self.deltaT
        #
        #     jointJerk = K.mean(K.sum(K.sum(K.square(adJerk), axis=-1), axis=-1), axis=-1)
        #
        #     jerkWeight = 0.7
        #
        #     return jointJerk * (1 - self.reconWeight) * jerkWeight + jointAcc * (1 - self.reconWeight) * (
        #                 1 - jerkWeight) + oloss * self.reconWeight
        #
        # elif elements[1] == 'bone':
        #
        #     boneLengthsLoss = self.boneLengthLoss(parents, adData, refBoneLengths)
        #     return boneLengthsLoss * (1 - self.reconWeight) + oloss * self.reconWeight

        elif elements[1] == 'acc-bone':

            jointAcc = self.accLoss(adData, refData)

            boneLengthsLoss = self.boneLengthLoss(self.classifier.trainloader.dataset.parents, adData, refBoneLengths)

            return boneLengthsLoss * (1 - self.reconWeight) * self.boneLenWeight + jointAcc * (1 - self.reconWeight) * (
                        1 - self.boneLenWeight) + oloss * self.reconWeight

    def unspecificAttack(self, labels):

        flabels = np.ones((len(labels), self.classifier.args.classNum))
        flabels = flabels * 1 / self.classifier.args.classNum

        return torch.LongTensor(flabels)

    def specifiedAttack(self, labels, targettedClasses=[]):
        if len(targettedClasses) <= 0:
            flabels = torch.LongTensor(np.random.randint(0, self.classifier.args.classNum, len(labels)))
        else:
            flabels = targettedClasses

        return flabels

    def abAttack(self, labels):

        flabels = labels

        return flabels

    def foolRateCal(self, rlabels, flabels, logits = None):

        hitIndices = []

        if self.attackType == 'ab':
            for i in range(0, len(flabels)):
                if flabels[i] != rlabels[i]:
                    hitIndices.append(i)
        elif self.attackType == 'abn':
            for i in range(len(flabels)):
                sorted,indices = torch.sort(logits[i], descending=True)
                ret = (indices[:self.topN] == rlabels[i]).nonzero(as_tuple=True)
                if len(ret) == 0:
                    hitIndices.append(i)

        elif self.attackType == 'sa':
            for i in range(0, len(flabels)):
                if flabels[i] == rlabels[i]:
                    hitIndices.append(i)

        return len(hitIndices) / len(flabels) * 100

    def getUpdate(self, grads, input):

        if self.updateRule == 'gd':
            self.learningRate = 0.01

            return input - grads * self.learningRate

        elif self.updateRule == 'Adam':
            if self.Adam == '':
                self.Adam = MyAdam()
            return self.Adam.get_updates(grads, input)

    def attack(self):

        self.classifier.model.eval()

        #set up the attack labels based on the attack type
        labels = self.classifier.trainloader.dataset.labels
        if self.attackType == 'abn':
            flabels = self.unspecificAttack(labels)
        elif self.attackType == 'sa':
            flabels = self.specifiedAttack(labels)
            oflabels = np.argmax(flabels, axis=-1)
        elif self.attackType == 'ab':
            flabels = self.abAttack(labels)
        else:
            print('specified targetted attack, no implemented')
            return

        for batchNo, (tx, ty) in enumerate(self.classifier.trainloader):
            adData = tx.clone()
            adData.requires_grad = True
            minCl = np.PINF
            maxFoolRate = np.NINF

            for ep in range(self.classifier.args.epochs):


                # compute the classification loss and gradient
                pred = self.classifier.model(adData)
                predictedLabels = torch.argmax(pred, axis=1)

                # computer the classfication loss gradient

                if self.attackType == 'ab':
                    classLoss = -torch.nn.CrossEntropyLoss()(pred, flabels[batchNo*self.classifier.args.batchSize:(batchNo+1)*self.classifier.args.batchSize])
                elif self.attackType == 'abn':
                    classLoss = torch.mean((pred - flabels[batchNo*self.classifier.args.batchSize:(batchNo+1)*self.classifier.args.batchSize])**2)
                else:
                    classLoss = torch.nn.CrossEntropyLoss()(pred, flabels[batchNo*self.classifier.args.batchSize:(batchNo+1)*self.classifier.args.batchSize])

                adData.grad = None
                classLoss.backward(retain_graph=True)
                cgs = adData.grad

                #computer the perceptual loss and gradient

                # computer the restBoneLengths
                positions = tx.reshape((tx.shape[0], tx.shape[1], -1, 3))

                boneVecs = positions - positions[:, :, self.classifier.trainloader.dataset.parents, :] + 1e-8

                boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))
                percepLoss = self.perceptualLoss(tx, adData, boneLengths)
                adData.grad = None
                percepLoss.backward(retain_graph=True)
                pgs = adData.grad


                if ep % 50 == 0:
                    print(f"Iteration {ep}/{self.classifier.args.epochs}, batchNo {batchNo}: Class Loss {classLoss:>9f}, Perceptual Loss: {percepLoss:>9f}")

                if self.attackType == 'ab':
                    foolRate = self.foolRateCal(ty, predictedLabels)
                elif self.attackType == 'abn':
                    foolRate = self.foolRateCal(ty, predictedLabels, pred)
                elif self.attackType == 'sa':
                    cFlabels = flabels[batchNo * self.classifier.args.batchSize:(batchNo + 1) * self.classifier.args.batchSize]
                    foolRate = self.foolRateCal(cFlabels, predictedLabels)
                else:
                    print('specified targetted attack, no implemented')
                    return

                if maxFoolRate < foolRate:
                    print('foolRate Improved! Iteration %d/%d, batchNo %d: Class Loss %.9f, Perceptual Loss: %.9f, Fool rate:%.2f' % (
                        ep, self.classifier.args.epochs, batchNo, classLoss, percepLoss, foolRate))
                    maxFoolRate = foolRate
                    folder = '/batch%d_%s_clw_%.2f_pl_%s_plw_%.2f/' % (
                        batchNo, self.attackType, self.classWeight, self.perpLossType, self.reconWeight)

                    if not os.path.exists(self.classifier.args.dataFolder + folder):
                        os.mkdir(self.classifier.args.dataFolder + folder)

                    if self.attackType == 'ab' or self.attackType == 'abn':
                        np.savez_compressed(
                            self.classifier.args.retFolder + folder + 'AdExamples_maxFoolRate_batch%d_AttackType_%s_clw_%.2f_pl_%s_reCon_%.2f_fr_%.2f.npz' % (
                                batchNo, self.attackType, self.classWeight, self.perpLossType, self.reconWeight, foolRate),
                            clips=adData.detach().numpy(), classes=predictedLabels.detach().numpy(),
                            oriClips=tx.detach().numpy(), tclasses=ty.detach().numpy(), classLos=classLoss.detach().numpy(),percepLoss=percepLoss.detach().numpy())
                    elif self.attackType == 'sa':
                        np.savez_compressed(
                            self.classifier.args.dataFolder + folder + 'AdExamples_maxFoolRate_batch%d_AttackType_%s_clw_%.2f_pl_%s_reCon_%.2f_fr_%.2f.npz' % (
                                batchNo, self.attackType, self.classWeight, self.perpLossType, self.reconWeight, foolRate),
                            clips=adData.detach().numpy(), classes=predictedLabels.detach().numpy(), fclasses=oflabels,
                            oriClips=tx.detach().numpy(),
                            tclasses=ty.detach().numpy(), classLos=classLoss.detach().numpy(),percepLoss=percepLoss.detach().numpy())

                if maxFoolRate == 100:
                    break;

                cgsnorms = K.sqrt(K.sum(K.square(cgs), axis=-1))

                cgsnorms = cgsnorms + 1e-18

                cgs = cgs / cgsnorms[:, :, np.newaxis]

                pgsnorms = K.sqrt(K.sum(K.square(pgs), axis=-1))

                pgsnorms = pgsnorms + 1e-18

                pgs = pgs / pgsnorms[:, :, np.newaxis]

                temp = self.getUpdate(cgs * self.classWeight + pgs * (1 - self.classWeight), adData)
                #temp = self.getUpdate(cgs, adData)

                missedIndices = []

                if self.attackType == 'ab':
                    for i in range(len(ty)):
                        if ty[i] == predictedLabels[i]:
                            missedIndices.append(i)
                elif self.attackType == 'abn':
                    for i in range(len(ty)):
                        sorted, indices = torch.sort(pred[i], descending=True)
                        ret = (indices[:self.topN] == ty[i]).nonzero(as_tuple=True)
                        if len(ret) > 0:
                            missedIndices.append(i)
                elif self.attackType == 'sa':
                    for i in range(len(ty)):
                        if cFlabels[i] != predictedLabels[i]:
                            missedIndices.append(i)

                tempCopy = adData.detach()
                if self.updateClip > 0:

                    updates = temp[missedIndices] - adData[missedIndices]
                    for ci in range(updates.shape[0]):
                        updateNorm = K.sqrt(K.sum(K.sum(K.square(updates[ci]))))

                        if updateNorm > self.updateClip:
                            updates[ci] = updates[ci] * self.updateClip / updateNorm

                    tempCopy[missedIndices] += updates
                else:
                    tempCopy[missedIndices] = temp[missedIndices]


