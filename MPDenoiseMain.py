from MPDenoise import WlrDenoiseModel, GradualWarmupScheduler
from MPDenoise import CharbonnierLoss, MixUp_AUG, torchPSNR, load_checkpoint
import numpy as np
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchvision import transforms
import torchvision.utils as vutils
from loadData import constructDatasetsWithFile, constructDatasetsWithFileNoEnhance, constructDatasetsWithDataNoEnhance
import matplotlib.pyplot as plt
from pylab import *
from torch.autograd import Function
import segyio
import pandas as pd
from torch.autograd import Variable
import h5py



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = 64

def savePlotSeis(path, data, w, h):
    cnt = 0
    col = max(int(h / w), 1)
    size = int(w * col)
    for j in range(int(len(data) / size)):
        clip = 1
        vmin, vmax = -clip, clip
        plt.figure(size, figsize=(10, 10))
        subplots_adjust(left=0.20, top=0.90, right=0.80, bottom=0.10, wspace=0.01, hspace=0.01)
        for i in range(size):
            plt.subplot(col, w, i + 1)
            plt.imshow(data[cnt], cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
            plt.axis('off')
            cnt += 1
    
        plt.savefig(path + str(j) + '.png')
  

############################# plot seis ################################

############################# 计算SNR ##################################
def SNR(data_origin, reconstructed):
    diff = data_origin - reconstructed
    down = np.sum(np.square(diff))
    up = np.sum(np.square(data_origin))
    snr = 10 * np.log10(up / down)
    return snr

############################# 计算SNR ##################################



############################# seismic #####################################
def computeErrCnt(noise):
    errCnt = 0
    for i in range(noise.shape[0]):
        errCnt += sum(abs(noise[i, :] > 0.15))
    return errCnt / (size * size)

def computeBatchErrCnt(output, noisy_data, batchSize):
    output = np.reshape(output, (batchSize, size, size))
    noisy_data = np.reshape(noisy_data, (batchSize, size, size))

    batchNoise = 0
    for i in range(batchSize):
        noise = noisy_data[i] - output[i]
        batchNoise += computeErrCnt(noise)

    return batchNoise / batchSize

def trainSeis():
    ####### Model ########
    model = WlrDenoiseModel(in_c=1)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)


    lr = 0.0002
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    OPTIM_NUM_EPOCHS = 300
    OPTIM_LR_MIN = 0.0002
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPTIM_NUM_EPOCHS - warmup_epochs + 40,
                                                            eta_min=OPTIM_LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Loss ###########
    l1_weights = 0.0
    criterion = CharbonnierLoss()

    ######### DataLoaders ###########
    start_epoch = 1
    batch_size = 32
    epoches = 500
    size = 160

    seis_path1 = ".\\data\\salt_and_overthrust_model\\SYNTHETIC_norm"
    seis_path2 = ".\\data\\synthesize\\data.sgy"
    
    train_dataset1 = constructDatasetsWithFile(seis_path1, size)
    train_dataset2 = constructDatasetsWithFile(seis_path2, size)

    train_dataset1 = np.ascontiguousarray(train_dataset1)
    train_dataset2 = np.ascontiguousarray(train_dataset2)


    train_dataset = np.vstack((train_dataset1, train_dataset2))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True)

    val_dataset = train_dataset1
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, epoches + 1))
    print('===> Loading datasets')

    noise_factor = 0.1
    best_psnr = 0
    best_epoch = 0
    best_iter = 0

    eval_now = len(train_loader) // 3 - 1
    print(f"\nEval after every {eval_now} Iterations !!!\n")
    mixup = MixUp_AUG()  ## 数据增强

    model_dir = '.\\modelPth'

    for epoch in range(start_epoch, epoches + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        model.train()
        for i, label in enumerate(train_loader, 0):
            # zero_grad
            for param in model.parameters():
                param.grad = None


            input_ = (label + noise_factor * torch.randn(*label.shape)).type(torch.FloatTensor).to(device)
            target = label.to(device)


            input_ = torch.unsqueeze(input_, dim=1)
            target = torch.unsqueeze(target, dim=1)

            restored = model(input_)


            output = restored.cpu().detach().numpy()
            #
            noisy_data = input_.cpu().detach().numpy()


            # Compute loss
            # 惩罚项
            l1_penalty = computeBatchErrCnt(output, noisy_data, batch_size)
            loss = (1 - l1_weights) * criterion(restored, target) + l1_weights * l1_penalty


            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            #### Evaluation ####
            if i % eval_now == 0 and i > 0 and (epoch in [1, 25, 45] or epoch > 60):
                model.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    data_val = np.reshape(data_val, (batch_size, 1,  size, size))
                    valInput = (data_val + noise_factor * np.random.randn(*data_val.shape)).type(torch.FloatTensor).to(device)
                    target = data_val.to(device)

                    with torch.no_grad():
                        restored = model(valInput)

                    for res, tar in zip(restored, target):
                        psnr_val_rgb.append(torchPSNR(res, tar))

                psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

                if psnr_val_rgb > best_psnr and math.isinf(psnr_val_rgb) == False:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join(model_dir, "model_seis_bestWithCAB.pth"))
                print("[epoch %d it %d PSNR: %.4f --- best_epoch %d best_iter %d Best_PSNR %.4f]" % (
                    epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))


                model.train()

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_seis_latestWithCAB.pth"))

def testSeis():
    noise_factor = 0.1
    batch_size = 1

    model = WlrDenoiseModel(in_c=1)
    weights_pth = '.\\modelPth\\model_seis_bestWithCAB.pth'
   
    load_checkpoint(model, weights_pth)
    print("===>Testing using weights: ", weights_pth)
    model = model.to(device)
    model.eval()

    seis_path = ".\\data\\synthesize\\data_test.sgy"

    test_dataset, w, h = constructDatasetsWithFileNoEnhance(seis_path, size)

    with torch.no_grad():
        origins = []
        outputs = []
        noises = []
        noisys = []
        db = 0
        snrs = 0
        i = 0
        maxSNR = -50
        for data in test_dataset:
            test_noisy = data

            db += SNR(data, test_noisy)

            label = torch.FloatTensor(data)
            label = torch.unsqueeze(label, dim=0)
            noisy = label + noise_factor * torch.randn(*label.shape)

            input = noisy.to(device)
            input = torch.unsqueeze(input, dim=0)

            restored_patch = model(input)

            output = restored_patch.cpu().detach().numpy()
            output = np.reshape(output, (size, size))
            noise_data = np.reshape(noisy, (size, size))
            noise = test_noisy - output

            snrs += SNR(data, output)
        
            origins.append(data)
            outputs.append(output)
            noises.append(noise)
            noisys.append(test_noisy)
        
            i += 1

        savePlotSeis('.\\fieldGussian3\\denoise', outputs, w, h)
        savePlotSeis('.\\fieldGussian3\\origins', origins, w, h)
        savePlotSeis('.\\fieldGussian3\\noisy', noisys, w, h)
        savePlotSeis('.\\fieldGussian3\\noises', noises, w, h)



if __name__ == '__main__':
    trainSeis()
    testSeis()













































































