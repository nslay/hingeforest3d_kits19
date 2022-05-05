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
import SimpleITK as sitk
import numpy as np
from RCCSeg import RCCSeg
from rcc_common import LoadImage, SaveImage
import functools
import operator

numFolds=10

# Threshold is in cm^3
def CleanUpMask(labelMap, threshold = 35):
    kidneyLabel = 1

    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    threshold *= 1000

    voxelVolume = functools.reduce(operator.mul, labelMap.GetSpacing())

    npLabelMap = sitk.GetArrayViewFromImage(labelMap)
    npNewLabelMap = npLabelMap.copy()

    ccMask = ccFilter.Execute(sitk.GetImageFromArray((npLabelMap > 0).astype(np.int16)))
    ccCount = ccFilter.GetObjectCount()

    print(f"Info: There are {ccCount} connected components.")

    npCcMask = sitk.GetArrayViewFromImage(ccMask)

    objectStats = []
    for ccLabel in range(1, ccCount+1):
        kidneyCount = (npLabelMap[npCcMask == ccLabel] == kidneyLabel).sum()
        objectStats.append((ccLabel, kidneyCount))

    objectStats.sort(reverse=True, key=lambda x : x[1])
    largestKidneyCcs = { obj[0] for obj in objectStats[:2] }

    for ccLabel in range(1, ccCount+1):
        voxelCount = (npCcMask == ccLabel).sum()
        volume = voxelCount * voxelVolume

        if volume < threshold:
            npNewLabelMap[npCcMask == ccLabel] = 0
        elif ccLabel not in largestKidneyCcs:
            npNewLabelMap[npCcMask == ccLabel] = 0

    newLabelMap = sitk.GetImageFromArray(npNewLabelMap)
    newLabelMap.CopyInformation(labelMap)

    return newLabelMap

def Resample(image, outputSpacing, interp):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputPixelType(sitk.sitkInt16)
    resampler.SetInterpolator(interp)

    newSize = (np.array(image.GetSize()) * np.array(image.GetSpacing()) / np.array(outputSpacing) + 0.5).astype(int)

    resampler.SetSize([int(newSize[0]), int(newSize[1]), int(newSize[2])])
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    return resampler.Execute(image)

def ResampleLow(image):
    outputSpacing = [ 2.0, 2.0, 2.0 ]
    interp = sitk.sitkNearestNeighbor
    return Resample(image, outputSpacing, interp)

def ResampleHigh(image):
    outputSpacing = [ 0.5, 0.5, 0.5 ]
    interp = sitk.sitkBSplineResamplerOrder3
    return Resample(image, outputSpacing, interp)

def GetBestSnapshot(snapshotDir):
    bestFile=os.path.join(snapshotDir, "bestDice.txt")

    bestEpoch=-1
    with open(bestFile, mode="rt", newline="") as f:
        line = next(iter(f))
        bestEpoch = int(line.split(" ")[-1])

    return os.path.join(snapshotDir, f"epoch_{bestEpoch}.pt")

def SegmentLowAndCrop(image, lowModelRoot, device="cpu"):
    #margin=0.1
    margin=0.2

    cad = RCCSeg()

    cad.SetDevice(device)

    npImage = sitk.GetArrayViewFromImage(image)

    imageLow = ResampleLow(image)

    npProbMapLow = None

    for f in range(numFolds):
        snapshotDir=os.path.join(lowModelRoot, f"snapshots_low_hingeforest_depth7_vggblock3_3d_fold{f+1}")

        snapshotFile = GetBestSnapshot(snapshotDir)

        print(f"Info: Loading {snapshotFile} ...")
        cad.LoadModel(snapshotFile)

        sitkProbMap, _ = cad.RunOne(imageLow)

        if npProbMapLow is None:
            npProbMapLow = sitk.GetArrayFromImage(sitkProbMap)
        else:
            npProbMapLow += sitk.GetArrayViewFromImage(sitkProbMap)

    npProbMapLow /= numFolds
    npLabelMapLow = npProbMapLow.argmax(axis=3).astype(np.int16)

    labelMapLow = sitk.GetImageFromArray(npLabelMapLow)
    labelMapLow.CopyInformation(imageLow)

    print(f"Info: Resampling label map {labelMapLow.GetSpacing()} --> {image.GetSpacing()}...")

    labelMap = Resample(labelMapLow, image.GetSpacing(), sitk.sitkNearestNeighbor)
    npLabelMap = sitk.GetArrayViewFromImage(labelMap)

    lower = np.argwhere(npLabelMap != 0).min(axis=0)
    upper = np.argwhere(npLabelMap != 0).max(axis=0)

    size = upper-lower

    print(f"Info: lower = {lower}, upper = {upper}, size = {size}, origSize = {npLabelMap.shape}")

    lower = np.maximum(0.0, (lower - margin*size)).astype(lower.dtype)
    upper = np.minimum(npImage.shape, lower + (1.0 + 2*margin)*size).astype(upper.dtype)

    npCroppedImage = npImage[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    npCroppedLabelMap = npLabelMap[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]

    newOrigin = image.TransformIndexToPhysicalPoint([ int(lower[2]), int(lower[1]), int(lower[0]) ])

    croppedImage = sitk.GetImageFromArray(npCroppedImage)
    croppedImage.SetSpacing(image.GetSpacing())
    croppedImage.SetDirection(image.GetDirection())
    croppedImage.SetOrigin(newOrigin)

    croppedLabelMap = sitk.GetImageFromArray(npCroppedLabelMap)
    croppedLabelMap.SetSpacing(image.GetSpacing())
    croppedLabelMap.SetDirection(image.GetDirection())
    croppedLabelMap.SetOrigin(newOrigin)

    return croppedImage, croppedLabelMap

def SegmentHighAndImpose(croppedImage, image, highModelRoot, device="cpu"):
    cad = RCCSeg()

    cad.SetDevice(device)

    imageHigh = ResampleHigh(croppedImage)

    npProbMapHigh = None

    for f in range(numFolds):
        snapshotDir=os.path.join(highModelRoot, f"snapshots_highcroppedspline_hingeforest_depth7_vggblock3_3d_fold{f+1}")

        snapshotFile = GetBestSnapshot(snapshotDir)

        print(f"Info: Loading {snapshotFile} ...")
        cad.LoadModel(snapshotFile)

        sitkProbMap, _ = cad.RunOne(imageHigh)

        if npProbMapHigh is None:
            npProbMapHigh = sitk.GetArrayFromImage(sitkProbMap)
        else:
            npProbMapHigh += sitk.GetArrayViewFromImage(sitkProbMap)

    npProbMapHigh /= numFolds
    npLabelMapHigh = npProbMapHigh.argmax(axis=3).astype(np.int16)

    labelMapHigh = sitk.GetImageFromArray(npLabelMapHigh)
    labelMapHigh.CopyInformation(imageHigh)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputPixelType(sitk.sitkInt16)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetReferenceImage(image)

    #return resampler.Execute(labelMapHigh)
    return CleanUpMask(resampler.Execute(labelMapHigh))

def main():
    dataRoot="/data/AIR/kits19/data"
    lowModelRoot="/data/AIR/kits19/Models"
    highModelRoot="/data/AIR/kits19/Models"
    outputRoot="/data/AIR/kits19/Output/Leaderboard_testing"
    listFile=os.path.join(dataRoot, "testing.txt")
    device="cuda:0"

    with open(listFile, mode="rt", newline="") as f:
        caseList = [ case.strip() for case in f if len(case.strip()) > 0 ]

    if not os.path.exists(outputRoot):
        os.makedirs(outputRoot)

    for case in caseList:
        imagePath = os.path.join(dataRoot, case, "imaging.nii.gz")

        print(f"\nInfo: Loading {imagePath} ...\n")
        image = sitk.ReadImage(imagePath)

        print(f"\nInfo: Running low resolution segmentation ...\n")
        croppedImage, croppedLabelMap = SegmentLowAndCrop(image, lowModelRoot, device=device)

        print(f"\nInfo: Running high resolution segmentation ...\n", flush=True)
        finalLabelMap = SegmentHighAndImpose(croppedImage, image, highModelRoot, device=device)

        cropOutputPath = os.path.join(outputRoot, f"{case}_cropped.nii.gz")
        outputPath = os.path.join(outputRoot, f"{case}_final.nii.gz")

        print(f"Info: Writing {cropOutputPath} ...")
        sitk.WriteImage(croppedLabelMap, cropOutputPath)

        print(f"Info: Writing {outputPath} ...")
        sitk.WriteImage(finalLabelMap, outputPath)

        

if __name__ == "__main__":
    main()

