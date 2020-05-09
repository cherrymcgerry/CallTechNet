import torch
import torch.nn as nn
import torch.optim as optim
from Model.InitModel import initModel
from Model.ConvNet1 import ConvNet
from Model.InitModel import saveCheckpoint
from Data.Setup_Database import setup_database
import xlsxwriter
from Model.ConvNet3 import cnn
from Model.ConvNet2 import ConvNetNoFC
import os
import pickle
import time

EPOCHS = 150
CHECKPOINT_FREQ = 5


class Model(object):
    def __init__(self, data_loader):
        print("setting up model")

        # setup test data
        self.testData_loader = setup_database(False, True, 20)

        # Setup device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.dict = {}
        self.totalDict = {}
        self.epoch = 0
        self.excelInitialized = False
        self.data_loader = data_loader
        self.loss = []
        self.result_root = './result'
        self.excel = []
        self.excelTotals = []
        self.maxVal = 0
        self.maxTrain = 0

        # inputI, output = next(iter(data_loader))
        # size = [inputI.size(), output.size()]
        #self.model = ConvNet()
        #self.model = ConvNetNoFC()
        self.model = cnn(3,64,0)

        self.optim = optim.Adam(self.model.parameters(), lr=0.001)

        initModel(self, self.device)

        # TODO EVAL


    def test(self):
        correct = 0
        total = 0
        self.model.eval()
        for i, data in enumerate(self.testData_loader):
            with torch.no_grad():

                inputI = data[0].view(-1, 3, 128, 128).to(device=self.device, dtype=torch.float)
                output = data[1].to(device=self.device, dtype=torch.float)

                prediction = self.model(inputI)

            # eval
            eval = []
            predicted_class = []
            real_class = []
            for sample in prediction:
                predicted_class.append(torch.argmax(sample))
            for sample in output:
                real_class.append(torch.argmax(sample))

            for i in range(len(predicted_class) ):
                eval.append({'pred': predicted_class[i], 'real': real_class[i]})

            for sample in eval:
                dictKeys = self.dict.keys()
                if sample['pred'] == sample['real']:
                    correct += 1
                    for key in dictKeys:
                        if sample['real'] == key:
                            self.dict[key] = {'correct': (self.dict.get(key)).get('correct'),
                                              'total': self.dict.get(key).get('total'),
                                              'testCorrect': self.dict.get(key).get('testCorrect') + 1,
                                              'testTotal': self.dict.get(key).get('testTotal') + 1}
                            break
                else:
                    for key in dictKeys:
                        if sample['real'] == key:
                            self.dict[key] = {'correct': (self.dict.get(key)).get('correct'),
                                              'total': self.dict.get(key).get('total'),
                                              'testCorrect': self.dict.get(key).get('testCorrect'),
                                              'testTotal': self.dict.get(key).get('testTotal') + 1}
                            break
                total += 1

        # self.epoch += 1
        print(F'Test Accuracy: {round(correct / total, 3)}')
        self.totalDict = { "trainAcc" : self.totalDict["trainAcc"], "testAcc" : round(correct / total, 3)}
        if round(correct/total,3) > self.maxVal:
            saveCheckpoint(self, 'maxValAccuracyCheckpoint.pth')
            self.maxVal = round(correct / total, 3)
            print('saving maxVal checkpoint')

        self.model.train()

    def train(self):
        lossF = nn.BCELoss()
        # lossF = nn.MSELoss()
        evalArr = []
        self.testTotals = 0
        self.testTotals2 = 0

        print("starting training loop")
        while self.epoch < EPOCHS:
            print(F'epochs: {self.epoch}/{EPOCHS}')
            correct = 0
            total = 0
            for i, data in enumerate(self.data_loader):
                self.model.zero_grad()

                #inputI = data[0].view(-1, 1, 128, 128).to(device=self.device, dtype=torch.float)
                #output = data[1].to(device=self.device, dtype=torch.float)
                inputI = data[0].type(torch.FloatTensor).to(self.device)
                output = data[1].to(self.device)

                start = time.time()

                prediction = self.model(inputI)[:,:,0,0]
                end = time.time()

                print(end-start)


                loss = lossF(prediction, output.float())

                loss.backward()
                self.optim.step()

                val, ind = prediction.max(1)
                val, trg = output.max(1)

                for target in trg:
                    if target == 10:
                        self.testTotals += 1


                # eval
                eval = []
                predicted_class = []
                real_class = []
                #for sample in prediction:
                #    predicted_class.append(torch.argmax(sample))
                #for sample in output:
                #    real_class.append(torch.argmax(sample))

                for i in range(len(ind)):
                    eval.append({'pred': ind[i], 'real': trg[i]})

                for sample in eval:
                    update = {}
                    if sample['real'] == 10:
                        self.testTotals2 += 1

                    dictKeys = self.dict.keys()
                    if sample['pred'] == sample['real']:
                        correct += 1
                        indict = False
                        for key in dictKeys:
                            if sample['real'] == key:
                                self.dict[key] = {'correct': ((self.dict.get(key)).get('correct') + 1),
                                                  'total': ((self.dict.get(key).get('total')) + 1), 'testCorrect': 0,
                                                  'testTotal': 0}
                                indict = True
                                break
                        if not indict:
                            update = {sample['real']: {'correct': 1, 'total': 1, 'testCorrect': 0, 'testTotal': 0}}
                            self.dict.update(update)

                    else:

                        indict = False
                        for key in dictKeys:
                            if sample['real'] == key:
                                self.dict[key] = {'correct': (self.dict.get(key)).get('correct'),
                                                  'total': ((self.dict.get(key).get('total')) + 1), 'testCorrect': 0,
                                                  'testTotal': 0}
                                indict = True
                                break

                        if not indict:

                            update = {sample['real']: {'correct': 0, 'total': 1, 'testCorrect': 0, 'testTotal': 0}}
                            self.dict.update(update)

                    total += 1

            self.epoch += 1
            # print(torch.cuda.current_device())
            # print(torch.cuda.get_device_name(torch.cuda.current_device()))
            # print(torch.cuda.is_available())
            print(f"totals of 10: {self.testTotals}")
            print(f"totals of 10: {self.testTotals2}")
            self.testTotals = 0
            self.testTotals2 = 0
            # accuracy + total, testaccuracy + total,
            print(F'Train Accuracy: {round(correct / total, 3)}')
            print(F'correct items {correct}')
            print(F'total items {total}')
            if round(correct / total, 3) > self.maxTrain:
                saveCheckpoint(self, 'maxTrainAccuracyCheckpoint.pth')
                self.maxTrain = round(correct / total, 3)
                print("saving maxtrainCheckpoint")
            if self.epoch % CHECKPOINT_FREQ == 0:
                saveCheckpoint(self, 'checkpoint.pth')
                self.dataToExcelTogether(F'resultsTogether{str(self.epoch/5)}.xlsx')
                self.dataToExcelSeparated(F'resultsSeparated{str(self.epoch/5)}.xlsx')
            self.totalDict = {"trainAcc" : round(correct / total, 3), "testAcc" : 0}
            self.test()
            actualTotal = 0
            for key, dict in sorted(self.dict.items()):
                print(
                    F'accuracy {key} : {round(dict["correct"] / dict["total"], 3)} total: {dict["total"]},  test: {round(dict["testCorrect"] / dict["testTotal"], 3)} testTotal: {dict["testTotal"]}')
                actualTotal += dict["total"]

            print(actualTotal)
            self.excel.append(self.dict)
            self.excelTotals.append(self.totalDict)

            self.dict = {}
            self.totalDict = {}
        saveCheckpoint(self, 'checkpointFinal.pth')
        self.dataToExcelTogether('resultsTogetherFinal.xlsx')
        self.dataToExcelSeparated('resultsSeperatedFinal.xlsx')
        print("Training finished")

    def getModel(self):
        return self.model

    def getEpoch(self):
        return self.epoch

    def dataToExcelSeparated(self, name):
        workbook = xlsxwriter.Workbook(name)
        worksheet = workbook.add_worksheet()
        column = 1
        row = 2

        worksheet = self.initializeExcel(worksheet)

        # write train data
        for i, sample in enumerate(self.excel):
            worksheet.write(row, column-1, i)
            for key, dict in sorted(sample.items()):
                worksheet.write(row, column, round(dict["correct"] / dict["total"], 3))
                column += 1
            #worksheet.write(row, column, dict["total"])
            column += 2
            for key, dict in sorted(sample.items()):
                worksheet.write(row, column, round(dict["testCorrect"] / dict["testTotal"], 3))
                column += 1
            column = 1
            row += 1
        workbook.close()

    def dataToExcelTogether(self, name):

        workbook = xlsxwriter.Workbook(name)
        worksheet = workbook.add_worksheet()
        with open(os.path.join('../101_ObjectCategories', 'label_dictionary.data'), 'rb') as f:
            labels = pickle.load(f)
        row = 1
        column = 1

        for label in labels:
            worksheet.write(row, column, label['label'])

            worksheet.write(row + 1, column, 'train')
            worksheet.write(row + 1, column + 1, 'test')
            column += 2
        worksheet.write(row,column, 'total')
        worksheet.write(row+1,column,'train')
        worksheet.write(row+1,column+1,'test')
        column = 1
        row = 3
        for i, sample in enumerate(self.excel):
            worksheet.write(row, column - 1, i)
            for key, dict in sorted(sample.items()):
                worksheet.write(0,column,dict["total"])
                worksheet.write(0,column+1,dict["testTotal"])
                worksheet.write(row, column, round(dict["correct"] / dict["total"], 3))
                worksheet.write(row, column+1, round(dict["testCorrect"] / dict["testTotal"], 3))
                column += 2
            worksheet.write(row, column, self.excelTotals[i]["trainAcc"])
            worksheet.write(row, column+1, self.excelTotals[i]["testAcc"])
            column = 1
            row += 1
        workbook.close()





    def initializeExcel(self, worksheet):

        with open(os.path.join('../101_ObjectCategories', 'label_dictionary.data'), 'rb') as f:
            labels = pickle.load(f)
        row = 0
        column = 1

        for label in labels:
            worksheet.write(row, column, label['label'])
            worksheet.write(row + 1, column, labels.index(label))
            column += 1

        column = len(labels) + 3
        for label in labels:
            worksheet.write(row, column, label['label'])
            worksheet.write(row + 1, column, labels.index(label))
            column += 1

        self.excelInitialized = True
        return worksheet
