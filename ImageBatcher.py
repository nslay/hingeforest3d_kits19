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
import random
import SimpleITK as sitk
import numpy as np
from HingeTree import contract
import torch
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

class EndEpochToken:
    pass

class ImageBatcher:
    def __init__(self, dataRoot, listFile, batchSize, numClasses=3, seed=6112, multipleOf=[16,16,16], window=455, level=75):
        ShowWarnings(False)
        self.dataRoot = dataRoot
        self.multipleOf = multipleOf
        self.numChannels = 1
        self.batchSize = batchSize
        self.numClasses = numClasses
        self.window = window
        self.level = level
        self.q = queue.Queue(maxsize=4)
        self.t = threading.Thread(target=self._loader_loop, args=(seed,))

        if isinstance(listFile, str):
            self._load_patient_ids(listFile)
        else:
            self.patientIds = listFile

    def start(self):
        if not self.t.is_alive():
            self.do_run = True
            self.t.start()

    def stop(self):
        while self.t.is_alive():
            self.do_run = False

            try:
                self.q.get_nowait()
            except:
                pass

            self.t.join(1.0)

    def _loader_loop(self, seed):
        random.seed(seed)

        while self.do_run:
            random.shuffle(self.patientIds)

            for i in range(0, len(self.patientIds), self.batchSize):
                if i + self.batchSize > len(self.patientIds):
                    tmpPatientIds = self.patientIds[i:]
                    tmpPatientIds += self.patientIds[:self.batchSize - len(tmpPatientIds)]
                else:
                    tmpPatientIds = self.patientIds[i:i + self.batchSize]

                if not self.do_run:
                    break

                volumePairs = self._load_patients(tmpPatientIds)

                if not self.do_run:
                    break

                npVolumes = [ pair[0] for pair in volumePairs ]
                npMasks = [ pair[1] for pair in volumePairs ]

                npImageBatch = np.concatenate(tuple(npVolumes), axis=0)
                npMaskBatch = np.concatenate(tuple(npMasks), axis=0)

                imageBatch = torch.from_numpy(npImageBatch).type(torch.float32)
                maskBatch = contract(torch.from_numpy(npMaskBatch[:, None, ...]), self.multipleOf).type(torch.long).squeeze(dim=1) # XXX: Magic!

                self.q.put((imageBatch, maskBatch))

            self.q.put(EndEpochToken()) 


    def __iter__(self):
        return self

    def __next__(self):
        value = self.q.get()

        if isinstance(value, EndEpochToken):
            raise StopIteration

        return value[0], value[1]

    def _load_patient_ids(self, listFile):
        self.patientIds = []

        with open(listFile, mode="rt", newline="") as f:
            self.patientIds = [ line.strip() for line in f if len(line.strip()) > 0 ]

    # Assume numpy convention
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

    def _crop_image_and_mask(self, npImg, npMask):
        if not self.cropKidney:
            return npImg, npMask

        indices = np.argwhere(npMask > 0)
        lower = indices.min(axis=0)
        upper = indices.max(axis=0)

        lower -= np.array(self.window_shape)
        upper += np.array(self.window_shape)

        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.array(npMask.shape))

        return npImg[..., lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]].copy(), npMask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]].copy()

    def _load_patients(self, patientIds):
        volumeMaskPairs = []

        def load_one_patient(patientId):
            imageFile = os.path.join(self.dataRoot, patientId, "imaging.nii.gz")
            maskFile = os.path.join(self.dataRoot, patientId, "segmentation.nii.gz")

            sitkVolume = sitk.ReadImage(imageFile) 
            sitkMask = sitk.ReadImage(maskFile)

            return sitkVolume, sitkMask

        with ThreadPoolExecutor(max_workers=min(8, self.batchSize)) as exe:
            volumeMaskPairs = list(exe.map(load_one_patient, patientIds))

        assert all(v is not None and m is not None for v, m in volumeMaskPairs)
        assert all(m.GetSize() == v.GetSize() for v, m in volumeMaskPairs)

        volumeMaskPairs = [ (sitk.GetArrayFromImage(v)[None, None, ...].astype(np.int16), sitk.GetArrayFromImage(m)[None, ...].astype(np.int16)) for v, m in volumeMaskPairs ]

        targetShape = np.array([ v[0].shape[2:] for v in volumeMaskPairs ]).min(axis=0)

        outputVolumeMaskPairs = []
        i = 0
        for npVolume, npMask in volumeMaskPairs:
            devZ = (npVolume.shape[2] - targetShape[0])//4
            devY = (npVolume.shape[3] - targetShape[1])//4
            devX = (npVolume.shape[4] - targetShape[2])//4

            # Random crops!
            beginZ = (npVolume.shape[2] - targetShape[0])//2 + random.randint(-devZ, devZ)
            endZ = beginZ + targetShape[0]
            beginY = (npVolume.shape[3] - targetShape[1])//2 + random.randint(-devY, devY)
            endY = beginY + targetShape[1]
            beginX = (npVolume.shape[4] - targetShape[2])//2 + random.randint(-devX, devX)
            endX = beginX + targetShape[2]

            """
            beginZ = random.randint(0, npVolume.shape[2] - targetShape[0])
            endZ = beginZ + targetShape[0]
            beginY = random.randint(0, npVolume.shape[3] - targetShape[1])
            endY = beginY + targetShape[1]
            beginX = random.randint(0, npVolume.shape[4] - targetShape[2])
            endX = beginX + targetShape[2]
            """

            npVolume = npVolume[..., beginZ:endZ, beginY:endY, beginX:endX]
            npMask = npMask[..., beginZ:endZ, beginY:endY, beginX:endX]

            npMask = self._resize_image(npMask)
            npVolume = self._resize_image(npVolume)

            npVolume = self._window_image(npVolume)

            outputVolumeMaskPairs.append((npVolume.copy(), npMask.copy()))

            i += 1

        return outputVolumeMaskPairs

if __name__ == "__main__":
    dataRoot="/data/AIR/kits19/data_high_cropped_spline"
    listFile=os.path.join(dataRoot, "..", "training.txt")

    batcher = ImageBatcher(dataRoot, listFile, 8, numClasses=3, seed=6112, multipleOf=[16,16,16])

    batcher.start()

    labelCounts = np.zeros([batcher.numClasses], dtype=np.int64)

    for batch in batcher:
        imageBatch, maskBatch = batch
        print(f"{type(imageBatch)}, {type(maskBatch)}")
        print(f"{imageBatch.shape}, {maskBatch.shape}")
        maskBatch = maskBatch.numpy()
        labelCounts += np.bincount(maskBatch[maskBatch >= 0], minlength=batcher.numClasses)

    print(labelCounts)
    weights = 1.0/labelCounts
    weights /= weights.max()
    print(weights)

    batcher.stop()

