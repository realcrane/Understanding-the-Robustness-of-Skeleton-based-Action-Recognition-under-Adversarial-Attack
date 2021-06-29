

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


# class HRNN(ActionRecogniser):
#     def __init__(self, batch_size=8, dataFolder='', retFolder='', classNum=0):
#         super().__init__(batch_size, dataFolder, retFolder)
#
#         self.classNum = classNum
#         self.model = ''
#         self.classWeight = 0.4
#         self.reconWeight = 0.6
#         self.boneLenWeight = 0.7
#         self.deltaT = 1 / 30
#         self.refBoneLengths = []
#         self.discriminator = ''
#         self.update_clip = 0
#         self.Adam = ''
#
#     def train(self, epochs, modelFile='', modelSaveFile=''):
#         data = np.load(self.dataFolder + 'classTrain.npz')
#
#         inputData = data['clips']
#         rlabels = data['classes']
#
#         labels = to_categorical(rlabels, self.classNum)
#
#         indices = [i for i in range(0, inputData.shape[0])]
#
#         indices = np.random.permutation(indices)
#
#         inputData = inputData[indices]
#         labels = labels[indices]
#
#         validBatchNo = 3
#         validSplit = int(len(inputData) / self.batch_size - validBatchNo) * self.batch_size
#
#         trainData = inputData[0: validSplit]
#         trainLabels = labels[0: validSplit]
#
#         validData = inputData[validSplit:validSplit + (validBatchNo * self.batch_size)]
#         validLabels = labels[validSplit:validSplit + (validBatchNo * self.batch_size)]
#
#         self.dataShape = inputData.shape
#
#         self.createModel()
#
#         if len(modelFile) > 0:
#             self.model.load_weights(self.retFolder + modelFile, by_name=True)
#
#         self.configureOptimiser()
#
#         self.classificationLoss()
#
#         self.model.compile(loss=self.classLoss, optimizer=self.optimiser)
#
#         minLossCheckpointer = ModelCheckpoint(self.retFolder + "minLoss_weights.hdf5", verbose=1, monitor='loss',
#                                               save_best_only=True, save_weights_only=True)
#         minValLossCheckpointer = ModelCheckpoint(self.retFolder + "minValLoss_weights.hdf5", verbose=1,
#                                                  monitor='val_loss', save_best_only=True, save_weights_only=True)
#
#         logger = CSVLogger(filename=self.retFolder + 'losses.log', append=False)
#
#         callback_list = [minLossCheckpointer, minValLossCheckpointer, logger]
#
#         self.model.fit(trainData, trainLabels, validation_data=(validData, validLabels), epochs=epochs,
#                        batch_size=self.batch_size, callbacks=callback_list)
#
#
#
#     def test(self, modelFile=''):
#
#         data = np.load(self.dataFolder + 'classTest.npz')
#
#         inputData = data['clips']
#         rlabels = data['classes']
#
#         labels = to_categorical(rlabels, self.classNum)
#
#         inputData = inputData[0: int(len(inputData) / self.batch_size) * self.batch_size]
#         labels = labels[0: int(len(labels) / self.batch_size) * self.batch_size]
#
#         self.dataShape = inputData.shape
#
#         self.createModel()
#
#         if len(modelFile) > 0:
#             self.model.load_weights(self.retFolder + modelFile, by_name=True)
#         else:
#             print('No model loaded.')
#             return
#
#         self.configureOptimiser()
#
#         self.classificationLoss()
#
#         self.model.compile(loss=self.classLoss, optimizer=self.optimiser)
#
#         rresults = self.model.predict(inputData, batch_size=self.batch_size)
#
#         results = np.argmax(rresults, axis=-1)
#
#         np.savetxt(self.retFolder + 'testRets.txt', results)
#         np.savetxt(self.retFolder + 'testGroundTruth.txt', rlabels)
#
#         # scores = self.model.evaluate(inputData, labels, batch_size=self.batch_size)
#
#         # print('%s: %.2f' % (self.model.metrics_names[0], scores))
#
#         hitIndices = []
#
#         for i in range(0, len(results)):
#             if rlabels[i] == results[i]:
#                 hitIndices.append(i)
#
#         hitIndices = np.array(hitIndices)
#
#         adData = inputData[hitIndices]
#         adLabels = rlabels[hitIndices]
#
#         print('Accurary: %.2f' % (len(hitIndices) / len(results)))
#         # validBatchNo = 3
#         # validSplit = int(len(adData) / self.batch_size - validBatchNo) * self.batch_size
#
#         # trainData = adData[0: validSplit]
#         # trainLabels = adLabels[0: validSplit]
#
#         # validData = adData[validSplit:validSplit+(validBatchNo*self.batch_size)]
#         # validLabels = adLabels[validSplit:validSplit+(validBatchNo*self.batch_size)]
#
#         np.savez_compressed(self.dataFolder + 'adClassTrain.npz', clips=adData, classes=adLabels)
#         # np.savez_compressed(self.dataFolder + 'adClassTest.npz', clips=validData, classes=validLabels)
#
#     def adTest(self, modelFile='', adExampleFile=''):
#
#         subFolders = [f for f in list(os.listdir(self.dataFolder)) if re.search('^batch', f) != None]
#
#         if len(adExampleFile) == 0:
#
#             for rfolder in subFolders:
#
#                 maxsa = 0
#                 maxus = 0
#                 maxab = 0
#
#                 elements = rfolder.split('_')
#
#                 at = elements[1]
#
#                 files = [f for f in list(os.listdir(self.dataFolder + '/' + rfolder))]
#
#                 for file in files:
#                     felements = os.path.splitext(file)[0].split('_')
#
#                     sr = felements[-1]
#                     if sr == 'rets':
#                         continue
#
#                     if at == 'ab':
#                         if float(sr) > maxab:
#                             maxab = float(sr)
#                             adExampleFile = file
#                     elif at == 'us':
#                         if float(sr) > maxus:
#                             maxus = float(sr)
#                             adExampleFile = file
#                     elif at == 'sa':
#                         if float(sr) > maxsa:
#                             maxsa = float(sr)
#                             adExampleFile = file
#                     else:
#                         print('Unknown attack type')
#
#                 print('Processing %s' % (self.dataFolder + '/' + rfolder + '/' + adExampleFile))
#                 data = np.load(self.dataFolder + '/' + rfolder + '/' + adExampleFile)
#
#                 inputData = data['clips']
#                 rlabels = data['classes']
#
#                 labels = to_categorical(rlabels, self.classNum)
#
#                 inputData = inputData[0: int(len(inputData) / self.batch_size) * self.batch_size]
#                 labels = labels[0: int(len(labels) / self.batch_size) * self.batch_size]
#
#                 self.dataShape = inputData.shape
#
#                 if self.model == '':
#                     self.createModel()
#
#                     if len(modelFile) > 0:
#                         self.model.load_weights(self.retFolder + modelFile, by_name=True)
#                     else:
#                         print('No model loaded.')
#                         return
#
#                     self.configureOptimiser()
#
#                     self.classificationLoss()
#
#                     self.model.compile(loss=self.classLoss, optimizer=self.optimiser)
#
#                 rresults = self.model.predict(inputData, batch_size=self.batch_size)
#
#                 np.savez_compressed(self.dataFolder + '/' + rfolder + '/' + adExampleFile + '_rets.npz',
#                                     rawRets=rresults, tlabels=data['tclasses'])
#
#         else:
#
#             data = np.load(self.dataFolder + adExampleFile)
#
#             inputData = data['clips']
#             rlabels = data['classes']
#
#             labels = to_categorical(rlabels, self.classNum)
#
#             inputData = inputData[0: int(len(inputData) / self.batch_size) * self.batch_size]
#             labels = labels[0: int(len(labels) / self.batch_size) * self.batch_size]
#
#             self.dataShape = inputData.shape
#
#             self.createModel()
#
#             if len(modelFile) > 0:
#                 self.model.load_weights(self.retFolder + modelFile, by_name=True)
#             else:
#                 print('No model loaded.')
#                 return
#
#             self.configureOptimiser()
#
#             self.classificationLoss()
#
#             self.model.compile(loss=self.classLoss, optimizer=self.optimiser)
#
#             rresults = self.model.predict(inputData, batch_size=self.batch_size)
#
#             np.savez_compressed(self.dataFolder + adExampleFile + '_rets.npz', rawRets=rresults,
#                                 tlabels=data['tclasses'])
#
#     def RNNLayerWithDropout(self, name='', input_dim=0, output_dim=0, length=0, dropoutRate=0, type='RNN', batchNorm=False,
#                             activation='tanh'):
#         m = '';
#
#         if type == 'LSTM':
#             m = Bidirectional(LSTM(units=int(output_dim), return_sequences=True, unroll=True),
#                               input_shape=(int(length), int(input_dim)), merge_mode='ave')
#         else:
#             m = Bidirectional(SimpleRNN(units=int(output_dim), return_sequences=True, unroll=True),
#                               input_shape=(int(length), int(input_dim)), merge_mode='ave')
#
#         if dropoutRate > 0:
#             m = dropout(dropoutRate)(m)
#
#         if batchNorm:
#             m = BatchNormalization(mode=2, axis=1)(m)
#
#         return m;
#
#     def createModel(self):
#
#         lLeg = [i for i in range(0, 15)]
#         rLeg = [i for i in range(15, 30)]
#         trunk = [i for i in range(30, 45)]
#         lArm = [i for i in range(45, 60)]
#         rArm = [i for i in range(60, 75)]
#
#         noiseInput = 0
#         hidden_dim = 128
#         shape = self.dataShape
#
#         lLegEncoder = self.RNNLayerWithDropout(name='lLegEncoder', input_dim=len(lLeg), output_dim=hidden_dim / 8,
#                                                length=shape[1])
#         rLegEncoder = self.RNNLayerWithDropout(name='rLegEncoder', input_dim=len(rLeg), output_dim=hidden_dim / 8,
#                                                length=shape[1])
#         trunkEncoder = self.RNNLayerWithDropout(name='trunkEncoder', input_dim=len(trunk), output_dim=hidden_dim / 8,
#                                                 length=shape[1])
#         lArmEncoder = self.RNNLayerWithDropout(name='lArmEncoder', input_dim=len(lArm), output_dim=hidden_dim / 8,
#                                                length=shape[1])
#         rArmEncoder = self.RNNLayerWithDropout(name='rArmEncoder', input_dim=len(rArm), output_dim=hidden_dim / 8,
#                                                length=shape[1])
#
#         ##Layer 2: L lower-body/trunk, R lower-body/trunk, L upper-body/trunk and R upper-body/trunk
#         lLowerTrunkEncoder = self.RNNLayerWithDropout(name='lLowerTrunkEncoder', input_dim=hidden_dim / 4,
#                                                       output_dim=hidden_dim / 4, length=shape[1])
#         rLowerTrunkEncoder = self.RNNLayerWithDropout(name='rLowerTrunkEncoder', input_dim=hidden_dim / 4,
#                                                       output_dim=hidden_dim / 4, length=shape[1])
#         lUpperTrunkEncoder = self.RNNLayerWithDropout(name='lUpperTrunkEncoder', input_dim=hidden_dim / 4,
#                                                       output_dim=hidden_dim / 4, length=shape[1])
#         rUpperTrunkEncoder = self.RNNLayerWithDropout(name='rUpperTrunkEncoder', input_dim=hidden_dim / 4,
#                                                       output_dim=hidden_dim / 4, length=shape[1])
#
#         ##Layer 3: Upper-body/Trunk, Lower-body/trunk
#
#         upperBodyTrunkEncoder = self.RNNLayerWithDropout(name='upperBodyTrunkEncoder', input_dim=hidden_dim / 2,
#                                                          output_dim=hidden_dim / 2, length=shape[1])
#         lowerBodyTrunkEncoder = self.RNNLayerWithDropout(name='lowerBodyTrunkEncoder', input_dim=hidden_dim / 2,
#                                                          output_dim=hidden_dim / 2, length=shape[1])
#
#         totalInput = Input(batch_shape=(self.batch_size, shape[1], shape[2]))
#         if noiseInput > 0:
#             noiseInputLayer = GaussianNoise(noiseInput, name='noiseInput')
#             noisedInput = noiseInputLayer(totalInput)
#
#         else:
#             noisedInput = totalInput
#
#         lLegInput = Lambda(lambda x: x[:, :, 0:15], name='lLegInputSplitter')(noisedInput)
#         rLegInput = Lambda(lambda x: x[:, :, 15:30], name='rLegInputSplitter')(noisedInput)
#         trunkInput = Lambda(lambda x: x[:, :, 30:45], name='trunkInputSplitter')(noisedInput)
#         lArmInput = Lambda(lambda x: x[:, :, 45:60], name='lArmInputSplitter')(noisedInput)
#         rArmInput = Lambda(lambda x: x[:, :, 60:75], name='rArmInputSplitter')(noisedInput)
#
#         ##encoding individual segments
#
#         encodedlLeg = lLegEncoder(lLegInput)
#
#         encodedrLeg = rLegEncoder(rLegInput)
#
#         encodedTrunk = trunkEncoder(trunkInput)
#
#         encodedlArm = lArmEncoder(lArmInput)
#
#         encodedrArm = rArmEncoder(rArmInput)
#
#         # combining them into larger parts
#         encodedlLowerTrunk = Concatenate(axis=-1)([encodedlLeg, encodedTrunk])
#
#         encodedlLowerTrunk = lLowerTrunkEncoder(encodedlLowerTrunk)
#
#         encodedrLowerTrunk = Concatenate(axis=-1)([encodedrLeg, encodedTrunk])
#
#         encodedrLowerTrunk = rLowerTrunkEncoder(encodedrLowerTrunk)
#
#         encodedlUpperTrunk = Concatenate(axis=-1)([encodedlArm, encodedTrunk])
#
#         encodedlUpperTrunk = lUpperTrunkEncoder(encodedlUpperTrunk)
#
#         encodedrUpperTrunk = Concatenate(axis=-1)([encodedrArm, encodedTrunk])
#
#         encodedrUpperTrunk = rUpperTrunkEncoder(encodedrUpperTrunk)
#
#         # combining them into upperbody and lowerbody
#         encodedLowerBodyTrunk = Concatenate(axis=-1)([encodedlLowerTrunk, encodedrLowerTrunk])
#
#         encodedLowerBodyTrunk = lowerBodyTrunkEncoder(encodedLowerBodyTrunk)
#
#         encodedUpperBodyTrunk = Concatenate(axis=-1)([encodedlUpperTrunk, encodedrUpperTrunk])
#
#         encodedUpperBodyTrunk = upperBodyTrunkEncoder(encodedUpperBodyTrunk)
#
#         input = Concatenate(axis=-1)([encodedLowerBodyTrunk, encodedUpperBodyTrunk])
#
#         LSTMRNNLayer = self.RNNLayerWithDropout(name='FullBodyEncoder', input_dim=hidden_dim, output_dim=hidden_dim,
#                                                 length=shape[1])
#
#         encoded_seq = LSTMRNNLayer(input)
#
#         encoded_seq = Flatten()(encoded_seq)
#
#         output = Dense(self.classNum, activation='softmax', name='softmax_layer')(encoded_seq)
#
#         model = Model(totalInput, output)
#
#         self.model = model
#
#     def configureOptimiser(self):
#
#         self.optimiser = Adam(lr=0.001, beta_1=0.9)
#
#     def classificationLoss(self):
#
#         self.classLoss = categorical_crossentropy
#
#     def adversarialLoss(self, y_pred, y_true):
#
#         return K.mean(K.sum(K.square(y_pred - y_true)))
#
#     def unspecificAttack(self, labels):
#
#         flabels = np.ones(labels.shape)
#         flabels = flabels * 1 / self.classNum
#
#         return flabels
#
#     def specifiedAttack(self, labels, targettedClasses=[]):
#
#         if len(targettedClasses) <= 0:
#             flabels = np.zeros(labels.shape)
#             for i in range(0, len(labels)):
#                 ind = np.argmax(labels[i])
#                 ind = (ind + np.random.randint(1, self.classNum - 1)) % self.classNum
#                 flabels[i][ind] = 1
#
#         else:
#
#             flabels = to_categorical(targettedClasses, self.classNum)
#
#         return flabels
#
#     def abAttack(self, labels):
#
#         flabels = labels
#
#         return flabels
#
#     def foolRateCal(self, rlabels, flabels):
#
#         hitIndices = []
#
#         if self.attackType == 'us' or self.attackType == 'ab':
#             for i in range(0, len(flabels)):
#                 if flabels[i] != rlabels[i]:
#                     hitIndices.append(i)
#         elif self.attackType == 'sa':
#             for i in range(0, len(flabels)):
#                 if flabels[i] == rlabels[i]:
#                     hitIndices.append(i)
#
#         return len(hitIndices) / len(flabels) * 100

