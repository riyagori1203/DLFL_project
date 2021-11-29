import test
from PyQt5 import QtCore, QtGui, QtWidgets
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random
import operator
import genres
# import music_genre


import math
import numpy as np
from collections import defaultdict

dataset = []
# import music_genre
# import test


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.file_path = ""
        self.trainingSet = []
        self.testSet = []

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 274)
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 0, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(30, 50, 331, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit.setText("")
        self.lineEdit.setCursorMoveStyle(QtCore.Qt.VisualMoveStyle)
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(380, 50, 93, 28))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(9)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(120, 100, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet(
            "background-color: rgb(255, 255, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(30, 160, 441, 51))
        font = QtGui.QFont()
        font.setFamily("Forte")
        font.setPointSize(10)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_2.setObjectName("lineEdit_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Music Genre Classifier"))
        self.lineEdit.setWhatsThis(_translate(
            "MainWindow", "<html><head/><body><p>Add path to file</p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Browse file"))
        self.pushButton_2.setText(_translate("MainWindow", "Predict Genre!"))
        self.lineEdit_2.setText(_translate("MainWindow", "Genre is: "))
        self.pushButton.clicked.connect(self.selectFile)
        self.pushButton_2.clicked.connect(self.predictg)

    def selectFile(self):
        self.file_path = (QtWidgets.QFileDialog.getOpenFileName())
        self.lineEdit.insert(self.file_path[0])

    # def loadDataset(genres):
    #     with open("my.dat", 'rb') as f:
    #         while True:
    #             try:
    #                 dataset.append(pickle.load(f))
    #             except EOFError:
    #                 f.close()
    #                 break

    # loadDataset("my.dat")

    # def distance(instance1, instance2, k, self):
    #     distance = 0
    #     mm1 = instance1[0]
    #     cm1 = instance1[1]
    #     mm2 = instance2[0]
    #     cm2 = instance2[1]
    #     distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    #     distance += (np.dot(np.dot((mm2-mm1).transpose(),
    #                                np.linalg.inv(cm2)), mm2-mm1))
    #     distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    #     distance -= k
    #     return distance

    # def getNeighbors(trainingSet, instance, k, self):
    #     distances = []
    #     for x in range(len(trainingSet)):
    #         dist = self.distance(trainingSet[x], instance, k) + \
    #             self.distance(instance, trainingSet[x], k)
    #         distances.append((trainingSet[x][2], dist))
    #     distances.sort(key=operator.itemgetter(1))
    #     neighbors = []
    #     for x in range(k):
    #         neighbors.append(distances[x][0])
    #     return neighbors

    # def nearestClass(neighbors, self):
    #     classVote = {}
    #     for x in range(len(neighbors)):
    #         response = neighbors[x]
    #         if response in classVote:
    #             classVote[response] += 1
    #         else:
    #             classVote[response] = 1
    #     sorter = sorted(classVote.items(),
    #                     key=operator.itemgetter(1), reverse=True)
    #     return sorter[0][0]

    # def predictg(self):
    #     i = 1
    #     results = defaultdict(int)
    #     for folder in os.listdir("C:/Users/Riya/Downloads/photometry-master/photometry-master/DLFL_project/genres/"):
    #         results[i] = folder
    #         i += 1

    #     (rate, sig) = wav.read(
    #         "photometry-master/DLFL_project/genres/pop/pop.00004.wav")
    #     mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    #     covariance = np.cov(np.matrix.transpose(mfcc_feat))
    #     mean_matrix = mfcc_feat.mean(0)
    #     feature = (mean_matrix, covariance, 0)

    #     pred = self.nearestClass(self.getNeighbors(dataset, feature, 5))

    #     print(results[pred])

    # # def distance(instance1, instance2, k):
    # #     distance = 0
    # #     mm1 = instance1[0]
    # #     cm1 = instance1[1]
    # #     mm2 = instance2[0]
    # #     cm2 = instance2[1]
    # #     distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    # #     distance += (np.dot(np.dot((mm2-mm1).transpose(),
    # #                                np.linalg.inv(cm2)), mm2-mm1))
    # #     distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    # #     distance -= k
    # #     return distance

    # # def getNeighbors(trainingSet, instance, k, self):
    # #     distances = []
    # #     for x in range(len(trainingSet)):
    # #         dist = self.distance(trainingSet[x], instance, k) + \
    # #             self.distance(instance, trainingSet[x], k)
    # #         distances.append((trainingSet[x][2], dist))
    # #     distances.sort(key=operator.itemgetter(1))
    # #     neighbors = []
    # #     for x in range(k):
    # #         neighbors.append(distances[x][0])
    # #     return neighbors

    # # def nearestClass(neighbors):
    # #     classVote = {}

    # #     for x in range(len(neighbors)):
    # #         response = neighbors[x]
    # #         if response in classVote:
    # #             classVote[response] += 1
    # #         else:
    # #             classVote[response] = 1

    # #     sorter = sorted(classVote.items(),
    # #                     key=operator.itemgetter(1), reverse=True)
    # #     return sorter[0][0]

    # # def getAccuracy(testSet, predictions):
    # #     correct = 0
    # #     for x in range(len(testSet)):
    # #         if testSet[x][-1] == predictions[x]:
    # #             correct += 1
    # #     return 1.0*correct/len(testSet)

    # # def loadDataset(filename, split, trSet, teSet):
    # #     dataset = []

    # #     with open("my.dat", 'rb') as f:
    # #         while True:
    # #             try:
    # #                 dataset.append(pickle.load(f))
    # #             except EOFError:
    # #                 f.close()
    # #                 break

    # #     for x in range(len(dataset)):
    # #         if random.random() < split:
    # #             trSet.append(dataset[x])
    # #         else:
    # #             teSet.append(dataset[x])

    # # def predictg(self):
    # #     directory = "C:/Users/Riya/Downloads/photometry-master/photometry-master/DLFL_project/genres/"
    # #     f = open("my.dat", 'wb')
    # #     i = 0

    # #     for folder in os.listdir(directory):
    # #         i += 1
    # #         if i == 11:
    # #             break
    # #         for file in os.listdir(directory+folder):
    # #             (rate, sig) = wav.read(directory+folder+"/"+file)
    # #             mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    # #             covariance = np.cov(np.matrix.transpose(mfcc_feat))
    # #             mean_matrix = mfcc_feat.mean(0)
    # #             feature = (mean_matrix, covariance, i)
    # #             pickle.dump(feature, f)

    # #     f.close()

    # #     trainingSet = []
    # #     testSet = []
    # #     self.loadDataset("my.dat", 0.66, trainingSet, testSet)

    # #     leng = len(testSet)
    # #     predictions = []
    # #     for x in range(leng):
    # #         predictions.append(self.nearestClass(
    # #             self.getNeighbors(trainingSet, testSet[x], 5)))

    # #     accuracy1 = self.getAccuracy(testSet, predictions)
    # #     print(accuracy1)

    # #     results = self.defaultdict(int)

    # #     i = 1
    # #     for folder in os.listdir("./photometry-master/DLFL_project/genres"):
    # #         results[i] = folder
    # #         i += 1

    # #     (rate, sig) = wav.read(
    # #         self.file_path)
    # #     mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    # #     covariance = np.cov(np.matrix.transpose(mfcc_feat))
    # #     mean_matrix = mfcc_feat.mean(0)
    # #     feature = (mean_matrix, covariance, 0)

    # #     pred = self.nearestClass(self.getNeighbors(self.dataset, feature, 5))

    # #     print(results[pred])

    dataset = []

    def loadDataset(filename):
        with open("my.dat", 'rb') as f:
            while True:
                try:
                    dataset.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break

    loadDataset("my.dat")

    def distance(self, instance1, instance2, k):
        distance = 0
        mm1 = instance1[0]
        cm1 = instance1[1]
        mm2 = instance2[0]
        cm2 = instance2[1]
        distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
        distance += (np.dot(np.dot((mm2-mm1).transpose(),
                                   np.linalg.inv(cm2)), mm2-mm1))
        distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
        distance -= k
        return distance

    def getNeighbors(self, trainingSet, instance, k):
        distances = []
        for x in range(len(trainingSet)):
            dist = self.distance(
                trainingSet[x], instance, k) + self.distance(instance, trainingSet[x], k)
            distances.append((trainingSet[x][2], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def nearestClass(self, neighbors):
        classVote = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVote:
                classVote[response] += 1
            else:
                classVote[response] = 1
        sorter = sorted(classVote.items(),
                        key=operator.itemgetter(1), reverse=True)
        return sorter[0][0]

    def predictg(self):
        results = defaultdict(int)

        i = 1
        for folder in os.listdir("./photometry-master/DLFL_project/genres"):
            results[i] = folder
            i += 1
        # C:/Users/Riya/Downloads/photometry-master/photometry-master/DLFL_project/genres/disco/disco.00003.wav
        filepath = self.convert(self.file_path)
        filepath = filepath.strip('All Files (*)')

        (rate, sig) = wav.read(
            filepath)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, 0)

        pred = self.nearestClass(self.getNeighbors(dataset, feature, 5))

        print(results[pred])

        self.lineEdit_2.insert(results[pred])

    def convert(self, file_path):
        st = ''.join(map(str, file_path))
        return st


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
