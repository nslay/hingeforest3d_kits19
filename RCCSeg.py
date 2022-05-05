# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# April 2022
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.ops as ops
import SimpleITK as sitk
import gc
import time
from ImageBatcher import ImageBatcher
from Net import Net
from DiceLoss import DiceLoss
from Deterministic import NotDeterministic
from HingeTree import expand, contract

class RCCSeg:
    def __init__(self, numClasses=3):
        self.device = "cpu"
        self.numClasses=numClasses
        self.multipleOf = [16,16,16]
        self.net = Net(in_channels=1,out_channels=self.numClasses, extra_outputs=self.multipleOf)
        self.dataRoot = None
        self.saveSteps = 25
        self.valSteps = 2*self.saveSteps
        self.dilateUnknown = False
        self.cropKidney = False
        self.level = 75
        self.window = 455

    def _get_roi_1d(self, size, multipleOf):
        remainder = (size % multipleOf)

        begin = int(remainder/2)
        end = begin + size - remainder

        return begin, end

    def _resize_image(self, npImg):
        beginX, endX = self._get_roi_1d(npImg.shape[-1], self.multipleOf[-1])
        beginY, endY = self._get_roi_1d(npImg.shape[-2], self.multipleOf[-2])
        beginZ, endZ = self._get_roi_1d(npImg.shape[-3], self.multipleOf[-3])

        return npImg[..., beginZ:endZ, beginY:endY, beginX:endX].copy()

    def _pad_image(self, npImg, shape):
        beginX, endX = self._get_roi_1d(shape[-1], self.multipleOf[-1])
        beginY, endY = self._get_roi_1d(shape[-2], self.multipleOf[-2])
        beginZ, endZ = self._get_roi_1d(shape[-3], self.multipleOf[-3])

        npImgOutput = np.zeros(shape, npImg.dtype)
        npImgOutput[..., beginZ:endZ, beginY:endY, beginX:endX] = npImg[...]

        return npImgOutput

    def _window_image(self, npImg):
        #return npImg
        perc = np.percentile(npImg, [0.5, 99.5])
        npImg = np.clip(npImg, perc[0], perc[1])
        #return ((npImg - npImg.mean())/(npImg.std() + 1e-10)).astype(np.float32)
        return npImg.astype(np.float32)

        """
        low = self.level - 0.5*self.window
        high = self.level + 0.5*self.window

        npImg = npImg.astype(np.float32)

        npImg[npImg < low] = low
        npImg[npImg > high] = high

        return ((npImg - low)/(high - low)).astype(np.float32)
        """

    def SetDevice(self, device):
        self.device = device
        self.net = self.net.to(device)

    def GetDevice(self):
        return self.device

    def SetDataRoot(self, dataRoot):
        self.dataRoot = dataRoot

    def GetDataRoot(self):
        return self.dataRoot

    def SaveModel(self, fileName):
        torch.save(self.net.state_dict(), fileName)

    def LoadModel(self, fileName):
        params = torch.load(fileName, map_location=self.GetDevice())
        #print(params.keys())
        #params["forest.linear_weights"] = params["forest.linear_weights"][[0,2], :]
        #params["forest.linear_bias"] = params["forest.linear_bias"][[0,2]]
        #params["forest.weights"] = self.net.forest.weights

        #params["forest.weights"] = self.net.forest.weights
        #params["forest.thresholds"] = self.net.forest.thresholds
        #params["forest.ordinals"] = self.net.forest.ordinals
        #params["forest.linear_weights"] = self.net.forest.linear_weights
        #params["forest.linear_bias"] = self.net.forest.linear_bias

        self.net.load_state_dict(params)

    def RunOne(self,patientId):
        if isinstance(patientId, str):
            volumePath=os.path.join(self.dataRoot, patientId, "imaging.nii.gz")
            sitkVolume = sitk.ReadImage(volumePath)
            npVolume = sitk.GetArrayViewFromImage(sitkVolume)[None, None, ...].astype(np.int16)
        elif isinstance(patientId, sitk.Image):
            sitkVolume = patientId
            npVolume = sitk.GetArrayViewFromImage(sitkVolume)[None, None, ...].astype(np.int16)
            patientId = None

        npVolume = self._window_image(npVolume)

        softmax = nn.Softmax(dim=1).to(self.GetDevice())

        with torch.no_grad():
            self.net.eval()

            batch = torch.from_numpy(self._resize_image(npVolume)).type(torch.float).to(self.GetDevice())
            npProbMap = self._pad_image(expand(softmax(self.net(batch)).cpu()).numpy(), [npVolume.shape[0], self.numClasses] + list(npVolume.shape[2:]))

            self.net.train()

        npProbMap = npProbMap.squeeze(axis=0).transpose(1,2,3,0)
        npLabelMap = npProbMap.argmax(axis=3).astype(np.int16)

        sitkProbMap = sitk.GetImageFromArray(npProbMap)
        sitkLabelMap = sitk.GetImageFromArray(npLabelMap)

        sitkProbMap.SetSpacing(sitkVolume.GetSpacing())
        sitkProbMap.SetDirection(sitkVolume.GetDirection())
        sitkProbMap.SetOrigin(sitkVolume.GetOrigin())

        sitkLabelMap.SetSpacing(sitkVolume.GetSpacing())
        sitkLabelMap.SetDirection(sitkVolume.GetDirection())
        sitkLabelMap.SetOrigin(sitkVolume.GetOrigin())

        return sitkProbMap, sitkLabelMap

    def Test(self,valList):
        if isinstance(valList, str):
            with open(valList, "rt", newline='') as f:
                patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]
        else:
            patientIds = valList

        allScores = None
        allLabels = None

        allDices = dict()
        npDices = np.zeros([len(patientIds), self.numClasses])

        for i, patientId in enumerate(patientIds):
            print(f"Info: Running '{patientId}' ...")

            maskFile = os.path.join(self.GetDataRoot(), patientId, "segmentation.nii.gz")

            gtMask = sitk.ReadImage(maskFile)
            npGtMask = sitk.GetArrayFromImage(gtMask).astype(np.int16)

            probMap, labelMap = self.RunOne(patientId)

            npProbMap = sitk.GetArrayFromImage(probMap)
            npProbMap = npProbMap[:,:,:,-1]

            npLabelMap = sitk.GetArrayFromImage(labelMap) # XXX: Very slow without making it numpy

            for label in range(1,self.numClasses):
                AintB, A, B = 0.0, 0.0, 0.0

                AintB += np.sum(np.logical_and((npGtMask == label), (npLabelMap == label)))
                A += np.sum(npGtMask == label)
                B += np.sum(npLabelMap == label)

                dice = 1.0 if A+B <= 0.0 else 2.0 * AintB / ( A + B )

                if patientId not in allDices:
                    allDices[patientId] = [ -1 ]*self.numClasses

                allDices[patientId][label] = dice
                npDices[i, label] = dice

                print(f"{label}: dice = {dice}")

        avgDice = [-1]*self.numClasses
        stdDice = [-1]*self.numClasses
        medDice = [-1]*self.numClasses

        for label in range(1,self.numClasses):
            npMask = (npDices[:,label] >= 0)

            if not npMask.any():
                continue

            avgDice[label] = npDices[npMask, label].mean()
            stdDice[label] = npDices[npMask, label].std()
            medDice[label] = np.median(npDices[npMask, label])

        return (avgDice, stdDice, medDice), allDices

    def Train(self,trainList,valList=None,valPerc=0.0,snapshotRoot="snapshots",startEpoch=0, seed=6112):
        batchSize=8
        labelWeights = torch.Tensor([1.0]*self.numClasses)
        numEpochs=3000

        print(f"Info: batchSize = {batchSize}")
        print(f"Info: numClasses = {self.numClasses}")
        print(f"Info: saveSteps = {self.saveSteps}")
        print(f"Info: valSteps = {self.valSteps}")
        print(f"Info: labelWeights = {labelWeights}")

        with open(trainList, mode="rt", newline="") as f:
            patientIds = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]

        if valList is None:
            if valPerc > 0.0:
                mid = max(1, int(valPerc*len(patientIds)))
                valList = patientIds[:mid]
                trainList = patientIds[mid:]
            else:
                trainList = patientIds
        else:
            trainList = patientIds

            with open(valList, mode="rt", newline="") as f:
                valList = [ patientId.strip() for patientId in f if len(patientId.strip()) > 0 ]

            print(f"Info: Loaded {len(valList)} validation cases.")

        imageBatcher = ImageBatcher(self.GetDataRoot(), trainList, batchSize, numClasses=self.numClasses, seed=seed, multipleOf=self.multipleOf, window=self.window, level=self.level)
        imageBatcher.start()

        criterion = nn.CrossEntropyLoss(ignore_index=-1,weight = labelWeights).to(self.GetDevice())
        criterion2 = DiceLoss(ignore_channel=0, ignore_label=-1, p=1, smooth=1e-3).to(self.GetDevice())

        optimizer = optim.Adam(self.net.parameters(), lr = 1e-3)

        trainLosses = np.ones([numEpochs])*1000.0

        if not os.path.exists(snapshotRoot):
            os.makedirs(snapshotRoot)

        for e in range(startEpoch,numEpochs):
            runningLoss = 0.0
            count = 0

            for xbatch, ybatch in imageBatcher:
                xbatch = xbatch.to(self.GetDevice())
                ybatch = ybatch.to(self.GetDevice())

                optimizer.zero_grad()            

                outputs = self.net(xbatch)

                with NotDeterministic():
                    loss1 = criterion(outputs, ybatch)
                    loss2 = criterion2(outputs, ybatch)

                    loss = loss1 + loss2

                    loss.backward()

                optimizer.step()

                runningLoss += loss
                count += 1

                print(f"loss = {loss.item()}, ce loss = {loss1.item()}, dice loss = {loss2.item()}", flush=True)

            gc.collect()
            torch.cuda.empty_cache() # Need this occassionally to prevent serious GPU memory fragmentation

            if count > 0:
                runningLoss /= count

            snapshotFile=os.path.join(snapshotRoot, f"epoch_{e}.pt")
            diceFile=os.path.join(snapshotRoot, f"dice_stats_{e}.txt")

            if ((e+1) % self.saveSteps) == 0:
                print(f"Info: Saving {snapshotFile} ...", flush=True)
                self.SaveModel(snapshotFile)
            else:
                print(f"Info: Skipping saving {snapshotFile}.", flush=True)

            trainLosses[e] = runningLoss

            if valList is None:
                print(f"Info: Epoch = {e}, training loss = {runningLoss}", flush=True)
            elif self.valSteps > 0 and ((e+1) % self.valSteps) == 0: 
                diceStats, allDices = self.Test(valList)
                print(f"Info: Epoch = {e}, training loss = {runningLoss}, validation dices = {diceStats[0]} +/- {diceStats[1]}", flush=True)

                with open(diceFile, mode="wt", newline="") as f:
                    for patientId in allDices:
                        f.write(f"{patientId}: {allDices[patientId]}\n")

                    f.write(f"\nDice stats: {diceStats[0]} +/- {diceStats[1]}\n")
 
        imageBatcher.stop()

        return trainLosses

