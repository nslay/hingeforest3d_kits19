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

import torch
import torch.nn as nn
import torch.nn.functional as F
from RandomHingeForest import RandomHingeForestFusedLinear
from HingeTree import expand

class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, extra_outputs=None):
        super().__init__()

        number_of_features=100
        number_of_trees=100
        depth=7

        print(f"number_of_features = {number_of_features}, number_of_trees = {number_of_trees}, depth = {depth}, extra_outputs = {extra_outputs}")

        self.conv11 = nn.Conv3d(in_channels, 40, 5, stride=4, padding=2, bias=False)
        self.bn11 = nn.BatchNorm3d(40, affine=False)
        self.conv12 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm3d(40, affine=False)
        self.conv13 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm3d(40, affine=False)

        self.pool = nn.MaxPool3d(2,2)

        self.conv21 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn21 = nn.BatchNorm3d(40, affine=False)
        self.conv22 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn22 = nn.BatchNorm3d(40, affine=False)
        self.conv23 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn23 = nn.BatchNorm3d(40, affine=False)

        self.conv31 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn31 = nn.BatchNorm3d(40, affine=False)
        self.conv32 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn32 = nn.BatchNorm3d(40, affine=False)
        self.conv33 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn33 = nn.BatchNorm3d(40, affine=False)

        self.features = nn.Conv3d(40, number_of_features, 1, bias=False)
        self.forestbn = nn.BatchNorm3d(number_of_features, affine=False)

        self.forest = RandomHingeForestFusedLinear(number_of_features, number_of_trees, out_channels, depth=depth, extra_outputs=extra_outputs)

    def calculate_features(self, x):
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = self.pool(x)

        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        #x = self.pool(x)

        x = self.forestbn(self.features(x))

        return x

    def forward(self, x):
        x = self.calculate_features(x)
        x = self.forest(x)

        return x

if __name__ == "__main__":
    device="cuda:0"
    net = Net().to(device)

    #x = torch.randn([8, 4, 80, 272, 176]).to(device)
    x = torch.randn([8, 1, 96, 160, 160]).to(device)

    x = net(x)

    x = x.cpu()
    x = expand(x)

    print(x.shape)

