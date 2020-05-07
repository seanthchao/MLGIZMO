import numpy as np
import os
# curFeature='Density'
curFeatureIndex={'Coordinates0': 0, 'Coordinates1': 1, 'Coordinates2': 2,\
                 'Velocities0': 3, 'Velocities1': 4, 'Velocities2': 5,\
                 'Density': 6, 'InternalEnergy': 7}

particleNum=4096
sampleNum=4096

num_input = particleNum
num_cls = 1

mlp = 32
num_sample = [num_input//4**(i+1) for i in range(10) if (num_input//4**(i+1))>12]
print(num_sample)

radius = [0.1, 0.2, 0.4, 0.8]
nn_uplimit = [16, 64, 64, 64]
channels = [[16, 16], [32, 32], [64, 64], [128, 128]]
multiplier = [[2,2], [2,2], [2,2], [2,2]]

assert(len(num_sample)==len(radius))
assert(len(num_sample)==len(nn_uplimit))
assert(len(num_sample)==len(channels))
assert(len(num_sample)==len(multiplier))

# =====================for final layer convolution=====================
global_channels = 512
global_multiplier = 2
# =====================================================================

weight_decay = 1e-5

kernel=[8,2,2]
binSize = np.prod(kernel)+1

normalize = False
pool_method = 'max'
nnsearch = 'sphere'
sample = 'FPS' #{'FPS','IDS','random'}

use_raw = True
with_bn = True
with_bias = False

# featureList=[['Density'], ['Coordinates', 0], ['Coordinates', 1], ['Coordinates', 2], \
#              ['Velocities', 0], ['Velocities', 1], ['Velocities', 2], \
#              ['InternalEnergy']]
featureList=[['Density']]


valList=['Coordinates', 'Velocities', 'Density', 'InternalEnergy']

rootDir=os.path.dirname(os.path.abspath(__file__))

# startFile=600; endFile=919  # train
# mode = 'training'
# dataDir='%s/Outputs/snapshots_p4e3_b1/s600e920sep001' %rootDir
startFile=3000; endFile=3000 # test
mode = 'evaluating'
dataDir='%s/Outputs/snapshots_p4e3_b1/s%03de%03dsep001' %(rootDir, startFile, endFile+1)
