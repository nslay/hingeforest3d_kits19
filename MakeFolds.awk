#!/usr/bin/awk -f

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

BEGIN {
  numFolds=10
  numCases=0
}

{
  cases[numCases++] = $0
}

END {

  beginVal=0
  for (f=0; f < numFolds; ++f) {
    endVal = int(numCases*(f+1)/numFolds)

    valFile="validation_fold" (f+1) ".txt"
    printf "" > valFile
    for (i = beginVal; i < endVal; ++i) 
      print cases[i] >> valFile

    close(valFile)

    trainFile="training_fold" (f+1) ".txt"
    printf "" > trainFile
    for (i = 0; i < beginVal; ++i)
      print cases[i] >> trainFile

    for (i = endVal; i < numCases; ++i)
      print cases[i] >> trainFile

    close(trainFile)

    beginVal = endVal
  }
}

