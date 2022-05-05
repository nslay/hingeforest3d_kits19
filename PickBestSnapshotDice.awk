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
    bestEpoch=-1
    bestDice=0.0
}

/Dice/ {
    epoch=-1
    if (match(FILENAME, /[0-9]+/) > 0)
        epoch=substr(FILENAME, RSTART, RLENGTH)
    else
        next

    if (match($0, /\[( *(-?[0-9]+\.?[0-9]+?),? *){3}\]/) > 0) {
        perfVector=substr($0, RSTART, RLENGTH)
        gsub(/[\[\] ]/, "", perfVector)
        #print perfVector
        delete tmp
        split(perfVector, tmp, ",")

        dice=tmp[length(tmp)] + 0.0
        #dice=tmp[2] + 0.0
        if (dice > bestDice) {
            bestEpoch = epoch
            bestDice = dice
        }
    }
    else
        next
}

END {
    print bestDice, bestEpoch
}

