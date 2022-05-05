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

resampler = sitk.ResampleImageFilter()
resampler.SetOutputPixelType(sitk.sitkInt16)

outputSpacing = [ 2.0, 2.0, 2.0 ]

dataRoot="/data/AIR/kits19/data"
destRoot="/data/AIR/kits19/data_low"

for caseDir in glob.glob(os.path.join(dataRoot, "case_*")):
    imagePath = os.path.join(caseDir, "imaging.nii.gz")
    segPath=os.path.join(caseDir, "segmentation.nii.gz")

    if not os.path.exists(segPath):
        continue

    image = sitk.ReadImage(imagePath)
    seg = sitk.ReadImage(segPath)

    print(image.GetSpacing())
    newSize = (np.array(image.GetSize()) * np.array(image.GetSpacing()) / np.array(outputSpacing) + 0.5).astype(int)

    resampler.SetSize([int(newSize[0]), int(newSize[1]), int(newSize[2])])
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    downImage = resampler.Execute(image)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    downSeg = resampler.Execute(seg)

    print(f"new size = {downImage.GetSize()}, new spacing = {downImage.GetSpacing()}")
    
    outputDir = os.path.join(destRoot, os.path.basename(caseDir))
    outputImagePath = os.path.join(outputDir, "imaging.nii.gz")
    outputSegPath = os.path.join(outputDir, "segmentation.nii.gz")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    print(f"Info: Writing {outputDir} ...")
    sitk.WriteImage(downImage, outputImagePath)
    sitk.WriteImage(downSeg, outputSegPath)

print("Done.")

