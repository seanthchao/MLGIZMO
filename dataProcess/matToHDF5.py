import scipy.io
import h5py as h5
import numpy as np
import pandas as pd

h5Pair=[1,2]
root='/media/gamer02/Data/sean/remoteTIARA_ML_SubGrid/EvrardCollapse'
matPath='%s/log_evrard_4X/Density/pred_votes%04d.mat' %(root, h5Pair[0])
hdf5Path='%s/Outputs/snapshots_p4e3_b1/s%03de%03dsep001' %(root, h5Pair[0], h5Pair[1])
h5PredOut='%s/log_evrard_4X/pred_votes%04d.hdf5' %(root, h5Pair[0])

# np.trim_zeros(mat['feature'][0])
# mat['pred'].shape
mat = scipy.io.loadmat(matPath)
# mat.keys()
for k in mat.keys():
    if '_' not in k:
        if mat[k].shape[0]>1:
            mat[k]=np.trim_zeros(mat[k][:,0])
        else:
            mat[k]=np.trim_zeros(mat[k][0])

h5PairFile=[h5.File('%s/snapshot_%04d.hdf5' %(hdf5Path, i), 'r') for i in h5Pair]

# Find IDs and corresponding Coords.
# dataDF=pd.DataFrame([])
IDPair=[h5PairFile[i]['PartType0']['ParticleIDs'][()] for i in range(2)]
newIndex=[np.where(IDPair[1]==i)[0][0] for i in IDPair[0]]
rhoPair=[h5PairFile[i]['PartType0']['Density'][()] for i in range(2)]

# [np.where(np.round(rhoPair[0], 4)==np.round(i, 4)) for i in mat['feature']]
dataDF=pd.DataFrame([IDPair[1], rhoPair[1]]).T
dataDF.columns=['ID', 'Rho']
dataDF.sort_values('Rho')
newIndexForFeature=np.array(dataDF['ID']).astype(int)

predDens=np.array([mat['pred'][i-1] for i in newIndexForFeature])
sudoCoord=np.array([h5PairFile[1]['PartType0']['Coordinates'][()][i-1] for i in newIndexForFeature])
sudoVel=np.array([h5PairFile[1]['PartType0']['Velocities'][()][i-1] for i in newIndexForFeature])
sudoSPHLength=np.array([h5PairFile[1]['PartType0']['SmoothingLength'][()][i-1] for i in newIndexForFeature])
sudoMass=np.array([h5PairFile[1]['PartType0']['Masses'][()][i-1] for i in newIndexForFeature])
sudoRho=np.array([h5PairFile[1]['PartType0']['Density'][()][i-1] for i in newIndexForFeature])

with h5.File(h5PredOut, 'w') as f:
    f.create_dataset('PartType0/Coordinates', dtype='float32'   , \
                    data=sudoCoord )
    f.create_dataset('PartType0/Velocities', dtype='float32'   , \
                    data=sudoVel )
    f.create_dataset('PartType0/SmoothingLength', dtype='float32'   , \
                    data=sudoSPHLength )
    f.create_dataset('PartType0/Density', dtype='float32'   , \
                    data=predDens )
    f.create_dataset('PartType0/DensityRes', dtype='float32'   , \
                    data=(predDens-sudoRho)/sudoRho )
    f.create_dataset('PartType0/Masses', dtype='float32'   , \
                    data=sudoMass )
    f.create_group('Header')
    f['Header'].attrs.create("NumPart_Total", [4096, 0, 0, 0, 0, 0], (6,), 'uint32')
    f['Header'].attrs.create("NumPart_ThisFile", [4096, 0, 0, 0, 0, 0], (6,), 'uint32')
    f['Header'].attrs.create("NumFilesPerSnapshot", 1, (1,), 'uint32')

    f.close()
