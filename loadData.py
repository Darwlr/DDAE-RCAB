import numpy as np
import segyio
import random
import torch
import h5py
from PIL import Image
import os
import cv2

def createH5Datasets(inputFile, outFile, size):
    allDatasets1 = constructGuassionNoiseDatasets(inputFile, size, 0.05)
    allDatasets3 = constructGuassionNoiseDatasets(inputFile, size, 0.15)
    allDatasets2 = constructSpNoiseDatasets(inputFile, size, 0.05)
    allDatasets4 = constructSpNoiseDatasets(inputFile, size, 0.1)

    hf = h5py.File(outFile, 'w')
    createSet(hf, 'train_set_1', 0, allDatasets1)
    createSet(hf, 'train_set_2', 1, allDatasets2)
    createSet(hf, 'train_set_3', 0, allDatasets3)
    createSet(hf, 'train_set_4', 1, allDatasets4)
    hf.close()

class MyDatasets():
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels
    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        return data, label
    def __len__(self):
        return len(self.datas)

def loadData(fileName):
    datas=[]
    with open(fileName) as txtData:
        lines = txtData.readlines()
        for line in lines:
            lineData=line.strip().split(',')    #去除空白和逗号“,”
            datas.append(lineData)
    return datas

def loadSgy(fileName):
    with segyio.open(fileName, ignore_geometry=True) as f:
        f.mmap()
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T
        return data

def loadFirstArrival(fileName):
    data = loadSgy(fileName)
    trainData = []

    traceNum = len(data[0])
    nsNum = len(data)

    tempNs = 0
    for i in range(traceNum):
        temp = []
        tempTrace = 0
        for i in range(nsNum):
            tempOneLine = data[tempTrace]
            temp.append(tempOneLine[tempNs])
            tempTrace += 1
        tempNs += 1
        trainData.append(temp)
    return trainData

def constructDatasetsWithCol(FileName, size):
    allData = loadSgy(FileName)
    allData = allData.flatten('F')  # 按列降维
    allDatasets = []
    for i in range(0, len(allData), size):
            tmp = allData[i: i + size]
            allDatasets.append(tmp)
            transposeTmp = np.fliplr(tmp)
            flipHorizontaTmp = np.flipud(tmp)
            allDatasets.append(transposeTmp)
            allDatasets.append(flipHorizontaTmp)
    return allDatasets.copy()
def constructDatasetsWithFile(FileName, size):
    allData = loadSgy(FileName)
    allDatasets = []
    for i in range(0, allData.shape[0] - size, size):
        for j in range(0, allData.shape[1] - size, size):
            tmp = allData[i:size + i, j:size + j]
            # if(np.sum(tmp) - 0.0 < 0.00001):
            #     continue
            allDatasets.append(tmp)
            transposeTmp = np.fliplr(tmp)
            flipHorizontaTmp = np.flipud(tmp)
            allDatasets.append(transposeTmp)
            allDatasets.append(flipHorizontaTmp)
    random.shuffle(allDatasets)
    return allDatasets.copy()
def constructDatasetsWithFileNoZero(FileName, size):
    allData = loadSgy(FileName)
    allDatasets = []
    for i in range(0, allData.shape[0] - size, size):
        for j in range(0, allData.shape[1] - size, size):
            tmp = allData[i:size + i, j:size + j]
            if(np.sum(tmp) - 0.0 < 0.1):
                continue
            allDatasets.append(tmp)
            transposeTmp = np.fliplr(tmp)
            flipHorizontaTmp = np.flipud(tmp)
            allDatasets.append(transposeTmp)
            allDatasets.append(flipHorizontaTmp)
    random.shuffle(allDatasets)
    return allDatasets.copy()

def constructDatasetsWithFileNoEnhance(FileName, size):
    allData = loadSgy(FileName)
    allDatasets = []
    w = 0
    h = 0
    for i in range(0, allData.shape[0] - size, size):
        h += 1
        for j in range(0, allData.shape[1] - size, size):
            w += 1
            tmp = allData[i:size + i, j:size + j]
            allDatasets.append(tmp)
    return allDatasets.copy(), int(w / h), h

def constructDatasetsWithDataNoEnhance(allData, size):
    allDatasets = []
    w = 0
    h = 0
    for i in range(0, allData.shape[0] - size, size):
        h += 1
        for j in range(0, allData.shape[1] - size, size):
            w += 1
            tmp = allData[i:size + i, j:size + j]
            allDatasets.append(tmp)
    return allDatasets.copy(), int(w / h), h

def constructDatasetsWithData(allData, size):
    allDatasets = []
    for i in range(0, allData.shape[0] - size, size):
        for j in range(0, allData.shape[1] - size, size):
            tmp = allData[i:size + i, j:size + j]
            allDatasets.append(tmp)
            transposeTmp = np.fliplr(tmp)
            flipHorizontaTmp = np.flipud(tmp)
            allDatasets.append(transposeTmp)
            allDatasets.append(flipHorizontaTmp)
    random.shuffle(allDatasets)
    return allDatasets

def constructGuassionNoiseDatasets(FileName, size, noise_factor):
    allData = loadSgy(FileName)
    allDatasets = []
    for i in range(0, allData.shape[0] - size, size):
        for j in range(0, allData.shape[1] - size, size):
            tmp = allData[i:size + i, j:size + j]
            tmp_noised = tmp + noise_factor * np.random.randn(*tmp.shape)
            allDatasets.append(tmp_noised)
            transposeTmp = np.fliplr(tmp)
            transposeTmp_noised = transposeTmp + noise_factor * np.random.randn(*transposeTmp.shape)
            flipHorizontaTmp = np.flipud(tmp)
            flipHorizontaTmp_noised = flipHorizontaTmp + noise_factor * np.random.randn(*flipHorizontaTmp.shape)
            allDatasets.append(transposeTmp_noised)
            allDatasets.append(flipHorizontaTmp_noised)
    random.shuffle(allDatasets)
    return allDatasets

def constructSpNoiseDatasets(FileName, size, noise_factor):
    allData = loadSgy(FileName)
    allDatasets = []
    for i in range(0, allData.shape[0] - size, size):
        for j in range(0, allData.shape[1] - size, size):
            tmp = allData[i:size + i, j:size + j]
            tmp_noised = addSpnoise(tmp, noise_factor)
            allDatasets.append(tmp_noised)
            transposeTmp = np.fliplr(tmp)
            transposeTmp_noised = addSpnoise(transposeTmp, noise_factor)
            flipHorizontaTmp = np.flipud(tmp)
            flipHorizontaTmp_noised = addSpnoise(flipHorizontaTmp, noise_factor)
            allDatasets.append(transposeTmp_noised)
            allDatasets.append(flipHorizontaTmp_noised)
    random.shuffle(allDatasets)
    return allDatasets

def constructMixNoiseDatasets(FileName, size, guss_prob, sp_prob):
    allData = loadSgy(FileName)
    allDatasets = []
    for i in range(0, allData.shape[0] - size, size):
        for j in range(0, allData.shape[1] - size, size):
            tmp = allData[i:size + i, j:size + j]
            tmp_noised = addMixNoise(tmp, guss_prob, sp_prob)
            allDatasets.append(tmp_noised)
            transposeTmp = np.fliplr(tmp)
            transposeTmp_noised = addMixNoise(transposeTmp, guss_prob, sp_prob)
            flipHorizontaTmp = np.flipud(tmp)
            flipHorizontaTmp_noised = addMixNoise(flipHorizontaTmp, guss_prob, sp_prob)
            allDatasets.append(transposeTmp_noised)
            allDatasets.append(flipHorizontaTmp_noised)
    random.shuffle(allDatasets)
    return allDatasets


# def constructPicDatasets(path, size):
#     datas = []
#     dirs = os.listdir(path)
#     for file in dirs:
#         path = os.path.join(path, file)

def getTrainLoader(batch_size, num_workers, allDatasets):
    # X_train = allDatasets.copy()
    train_loader = torch.utils.data.DataLoader(allDatasets.copy(), batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    return train_loader
def getLoader(batch_size, num_workers, allDatasets, train_size=0.6):
    X_train = allDatasets[: int(len(allDatasets) * train_size)]
    X_val = allDatasets[int(len(allDatasets) * train_size): ]
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    return train_loader, val_loader

#输入为list类型数据,分割为特征和标签两部分，返回为np.narray类型的特征数组和标签数组
def splitData(dataSet):
    character=[]
    label=[]
    for i in range(len(dataSet)):
        character.append([float(tk) for tk in dataSet[i][:-1]])
        label.append(dataSet[i][-1])
    return np.asarray(character, dtype=float), np.asarray(label, dtype=float)

def MaxNormalization(data):
    absData = np.abs(data)
    maxV = np.max(absData, axis=0)
    return data / maxV

def MaxNormalizationWhole(data):
    absData = np.abs(data)
    maxlineV = np.max(absData, axis=0)
    maxV = np.max(maxlineV, axis=0)
    print("max = {}".format(maxV))
    return data / maxV

def addgussian(data, prob = 0.15):
    data_noised = data.copy()
    data_noised = (data_noised + prob * np.random.randn(*data_noised.shape))
    return data_noised

def addSpnoise(data, prob = 0.05):
    if(type(data) == np.ndarray):
        output = data.copy()
    else:
        output = data.clone()
    thres = 1 - prob
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            rand = random.random()
            if rand < prob:
                output[i][j] = 0
            elif rand > thres:
                output[i][j] = 255
    return output

def addSpnoise2(data, snr):
    if(type(data) == np.ndarray):
        output = data.copy()
    else:
        output = data.clone()
    sp = data.shape[0] * data.shape[1]
    NP = int(sp*(1-snr))
    for i in range(NP):
        randx = random.randint(1, data.shape[0] - 1)
        randy = random.randint(1, data.shape[1] - 1)
        if random.random() <= 0.5:
            output[randx, randy] = 0
        else:
            output[randx, randy] = 255
    return output

def addTensorSpnoise(data, prob = 0.05):
    output = data.clone()
    thres = 1 - prob
    for k in range(data.shape[0]):
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                rand = random.random()
                if rand < prob:
                    output[k, 0, i, j] = 0
                elif rand > thres:
                    output[k, 0, i, j] = 255
                else:
                    output[k, 0, i, j] = data[k, 0, i, j]
    return output

def addMixNoise(data, guss_prob, sp_prob):
    output = addSpnoise(addgussian(data, guss_prob), sp_prob)
    return output

def createSet(hf, name, tip, data):
    hf.create_dataset(name, data=data)
    t = [[tip] * len(data)]
    hf.create_dataset(name + '_label', data=t)

def get_files(fileName):
    f = h5py.File(fileName, 'r')
    X_train_1 = f['train_set_1']
    Y_train_1 = f['train_set_1_label']

    X_train_2 = f['train_set_2']
    Y_train_2 = f['train_set_2_label']

    X_train_3 = f['train_set_3']
    Y_train_3 = f['train_set_3_label']

    X_train_4 = f['train_set_4']
    Y_train_4 = f['train_set_4_label']

    X_train_5 = f['train_set_5']
    Y_train_5 = f['train_set_5_label']

    X_train_6 = f['train_set_6']
    Y_train_6 = f['train_set_6_label']


    # 把所有数据集进行合并
    image_list = np.vstack((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6))
    label_list = np.hstack((Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5, Y_train_6))

    return image_list, label_list

def get_files_noise(fileName):
    f = h5py.File(fileName, 'r')
    X_train_1 = f['train_set_1']
    Y_train_1 = f['train_set_1_label']

    X_train_2 = f['train_set_2']
    Y_train_2 = f['train_set_2_label']

    X_train_3 = f['train_set_3']
    Y_train_3 = f['train_set_3_label']

    X_train_4 = f['train_set_4']
    Y_train_4 = f['train_set_4_label']

    # 把所有数据集进行合并
    image_list = np.vstack((X_train_1, X_train_2, X_train_3, X_train_4))
    label_list = np.hstack((Y_train_1, Y_train_2, Y_train_3, Y_train_4))

    return image_list, label_list
