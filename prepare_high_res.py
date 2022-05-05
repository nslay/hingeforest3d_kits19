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
import glob

margin=0.1

resampler = sitk.ResampleImageFilter()
resampler.SetOutputPixelType(sitk.sitkInt16)

outputSpacing = [ 0.5, 0.5, 0.5 ]

dataRoot="/data/AIR/kits19/data"
destRoot="/data/AIR/kits19/data_high_cropped_spline"

for caseDir in glob.glob(os.path.join(dataRoot, "case_*")):
    imagePath = os.path.join(caseDir, "imaging.nii.gz")
    segPath=os.path.join(caseDir, "segmentation.nii.gz")

    if not os.path.exists(segPath):
        continue

    image = sitk.ReadImage(imagePath)
    seg = sitk.ReadImage(segPath)

    npImage = sitk.GetArrayViewFromImage(image)
    npSeg = sitk.GetArrayViewFromImage(seg)

    lower = np.argwhere(npSeg != 0).min(axis=0)
    upper = np.argwhere(npSeg != 0).max(axis=0)

    size = upper-lower

    lower = np.maximum(0.0, (lower - margin*size)).astype(lower.dtype)
    #upper = np.minimum(npImage.shape, (upper + margin*size)).astype(upper.dtype)
    upper = np.minimum(npImage.shape, lower + (1.0 + 2*margin)*size).astype(upper.dtype)

    npCroppedImage = npImage[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    npCroppedSeg = npSeg[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]

    newOrigin = image.TransformIndexToPhysicalPoint([ int(lower[2]), int(lower[1]), int(lower[0]) ])

    croppedImage = sitk.GetImageFromArray(npCroppedImage)
    croppedSeg = sitk.GetImageFromArray(npCroppedSeg)

    croppedImage.SetSpacing(image.GetSpacing())
    croppedImage.SetDirection(image.GetDirection())
    croppedImage.SetOrigin(newOrigin)

    croppedSeg.SetSpacing(image.GetSpacing())
    croppedSeg.SetDirection(image.GetDirection())
    croppedSeg.SetOrigin(newOrigin)

    print(image.GetSpacing())
    newSize = (np.array(croppedImage.GetSize()) * np.array(image.GetSpacing()) / np.array(outputSpacing) + 0.5).astype(upper.dtype)

    resampler.SetSize([int(newSize[0]), int(newSize[1]), int(newSize[2])])
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputDirection(croppedImage.GetDirection())
    resampler.SetOutputOrigin(croppedImage.GetOrigin())

    resampler.SetInterpolator(sitk.sitkBSplineResamplerOrder3)
    upCroppedImage = resampler.Execute(croppedImage)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    upCroppedSeg = resampler.Execute(croppedSeg)

    print(f"new size = {upCroppedImage.GetSize()}, new spacing = {upCroppedImage.GetSpacing()}")
    
    outputDir = os.path.join(destRoot, os.path.basename(caseDir))
    outputImagePath = os.path.join(outputDir, "imaging.nii.gz")
    outputSegPath = os.path.join(outputDir, "segmentation.nii.gz")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    print(f"Info: Writing {outputDir} ...")
    sitk.WriteImage(upCroppedImage, outputImagePath)
    sitk.WriteImage(upCroppedSeg, outputSegPath)

print("Done.")

