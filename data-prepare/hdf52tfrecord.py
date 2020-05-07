'''
For each snapshot:
    Serial:
        1.  Splitting (NPart-1) data randomly into [train, test, val] list
            -> Create evrard_[train, test, val]_list.txt storing file path
        2.  Create [train, test, val, train_val]_files.txt storing file paths named:*.tfrecord
        (Or convert above procedures to a random process from the beginning??)
        3.  Global dataset for [u, h, rho], which are invariant from different particle's
            point of view
    Parallel:
        2.  Do the data labelling and augmentation.
            Labelling includes:
                Density:[L:0, H:1]
            Augmentation includes:
                Change reference particle: *(NPart-1)
        3.  HDF5 -> TXT -> tfrecord (or even HDF5 -> tfrecord !?)
'''

import h5py as h5
import numpy as np
import tensorflow as tf
import os, sys, random, argparse, imp
from multiprocessing import Pool

# global dataDir, valList, tRhoMean, randomList, store_folder, oriList, rho_t, ids
global fileDict, testfile, trainfile, curFeature, MPI_Inside, targetH5File_part

def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    # print(out_str)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def moveToOrigin(xyz, center):
    return xyz-center

def makeTFRecords_part(part):
    f=[h5.File('%s/snapshot_%04d.hdf5' %(Config.dataDir, targetH5File_part)  , 'r'), \
       h5.File('%s_next/snapshot_%04d.hdf5' %(Config.dataDir, targetH5File_part+1), 'r')]


    # prepare training data
    valDict={}; targetID=[]
    trainDict={}
    gas=[f[i]['PartType0'] for i in range(2)]
    ids=[np.array(gas[i]['ParticleIDs'][()]) for i in range(2)]
    for val in Config.valList:
        valDict[val]=[np.array(gas[i][val][()]) for i in range(2)]
    originVec={}
    # part = 2508
    targetID.append(part)
    targetID.append(np.where(ids[1]==ids[0][part])[0][0])
    for vec in ['Coordinates', 'Velocities']:
        originVec[vec]=valDict[vec][0][targetID[0]] # respect to the vec at t=0.
        valDict[vec][0] = np.array([moveToOrigin(j, originVec[vec]) for j in valDict[vec][0]])

    phase=np.random.choice(2, 1, p=[0.8, 0.2])[0]
    filename = 'evrard_%03d_%06d.tfrecord'%(targetH5File_part, part)
    filepath = os.path.join(store_folder, filename)

    if Config.mode == 'evaluating':
        testfile = open(os.path.join(store_folder, 'test_files.txt'), 'a')
        filepath = os.path.join(store_folder, filename)
        testfile.write("%s\n"%filepath)
        testfile.close()
    elif Config.mode == 'training':
        if phase == 0 :
            trainfile = open(os.path.join(store_folder, 'train_files.txt'), 'a')
            filepath = os.path.join(store_folder, filename)
            trainfile.write("%s\n"%filepath)
            trainfile.close()
        else:
            testfile = open(os.path.join(store_folder, 'test_files.txt'), 'a')
            filepath = os.path.join(store_folder, filename)
            testfile.write("%s\n"%filepath)
            testfile.close()

    if len(curFeature) > 1:
        feature = valDict[curFeature[0]][1][targetID[1]][curFeature[1]]
    else:
        feature = valDict[curFeature[0]][1][targetID[1]]

    filepath = os.path.join(store_folder, filename)

    xyz = valDict['Coordinates'][0]
    xyz /= np.sqrt(np.mean(np.sum(np.square(xyz), axis=1)))
    xyz_vel = valDict['Velocities'][0]
    xyz_vel /= np.sqrt(np.mean(np.sum(np.square(xyz_vel), axis=1)))
    rho = valDict['Density'][0]
    rho /= np.sqrt(np.mean(np.square(rho)))
    engy = valDict['InternalEnergy'][0]
    engy /= np.sqrt(np.mean(np.square(engy)))

    writer = tf.io.TFRecordWriter(filepath)

    xyz_raw = xyz.tostring()
    xyz_vel_raw = xyz_vel.tostring()
    rho_raw = rho.tostring()
    engy_raw = engy.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'xyz_vel_raw': _bytes_feature(xyz_vel_raw),
        'rho_raw': _bytes_feature(rho_raw),
        'engy_raw': _bytes_feature(engy_raw),
        'feature': _float_feature(feature),
        'xyz_raw': _bytes_feature(xyz_raw)}))
    writer.write(example.SerializeToString())

    for i in range(2):
        f[i].close()

def makeTFRecords(targetH5File):
    partList=fileDict[targetH5File]
    f=[h5.File('%s/snapshot_%04d.hdf5' %(Config.dataDir, targetH5File)  , 'r'), \
       h5.File('%s_next/snapshot_%04d.hdf5' %(Config.dataDir, targetH5File+1), 'r')]

    # prepare training data
    valDict={}; targetID=[]
    trainDict={}
    gas=[f[i]['PartType0'] for i in range(2)]
    ids=[np.array(gas[i]['ParticleIDs'][()]) for i in range(2)]
    for val in Config.valList:
        valDict[val]=[np.array(gas[i][val][()]) for i in range(2)]
    for part in partList:
        originVec={}
        targetID.append(part)
        targetID.append(np.where(ids[1]==ids[0][part])[0][0])
        for vec in ['Coordinates', 'Velocities']:
            originVec[vec]=valDict[vec][0][targetID[0]] # respect to the vec at t=0.
            valDict[vec][0] = np.array([moveToOrigin(j, originVec[vec]) for j in valDict[vec][0]])

        phase=np.random.choice(2, 1, p=[0.8, 0.2])[0]
        filename = 'evrard_%03d_%06d.tfrecord'%(targetH5File, part)
        filepath = os.path.join(store_folder, filename)

        if Config.mode == 'evaluating':
            testfile = open(os.path.join(store_folder, 'test_files.txt'), 'a')
            filepath = os.path.join(store_folder, filename)
            testfile.write("%s\n"%filepath)
            testfile.close()
        elif Config.mode == 'training':
            if phase == 0 :
                trainfile = open(os.path.join(store_folder, 'train_files.txt'), 'a')
                filepath = os.path.join(store_folder, filename)
                trainfile.write("%s\n"%filepath)
                trainfile.close()
            else:
                testfile = open(os.path.join(store_folder, 'test_files.txt'), 'a')
                filepath = os.path.join(store_folder, filename)
                testfile.write("%s\n"%filepath)
                testfile.close()

        if len(curFeature) > 1:
            feature = valDict[curFeature[0]][1][targetID[1]][curFeature[1]]
        else:
            feature = valDict[curFeature[0]][1][targetID[1]]

        filepath = os.path.join(store_folder, filename)

        xyz = valDict['Coordinates'][0]
        xyz /= np.sqrt(np.mean(np.sum(np.square(xyz), axis=1)))
        xyz_vel = valDict['Velocities'][0]
        xyz_vel /= np.sqrt(np.mean(np.sum(np.square(xyz_vel), axis=1)))
        rho = valDict['Density'][0]
        rho /= np.sqrt(np.mean(np.square(rho)))
        engy = valDict['InternalEnergy'][0]
        engy /= np.sqrt(np.mean(np.square(engy)))

        writer = tf.io.TFRecordWriter(filepath)

        xyz_raw = xyz.tostring()
        xyz_vel_raw = xyz_vel.tostring()
        rho_raw = rho.tostring()
        engy_raw = engy.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'xyz_vel_raw': _bytes_feature(xyz_vel_raw),
            'rho_raw': _bytes_feature(rho_raw),
            'engy_raw': _bytes_feature(engy_raw),
            'feature': _float_feature(feature),
            'xyz_raw': _bytes_feature(xyz_raw)}))
        writer.write(example.SerializeToString())

        for i in range(2):
            f[i].close()

if len(sys.argv) < 2:
    print('Please provide the Config file.'); sys.exit(0)

Config = imp.load_source("module.name", sys.argv[1])

fileDict={}
particleList=range(0,(Config.endFile-Config.startFile+1)*Config.particleNum)
samplePartList=random.sample(particleList, Config.sampleNum )

for i in samplePartList:
    fileNum=i//Config.particleNum+Config.startFile
    if fileNum not in list(fileDict.keys()):
        fileDict[fileNum]=[]
    fileDict[fileNum].append(i%Config.particleNum)

for i in Config.featureList:
    curFeature=i
    if len(i)>1:
        store_folder='%s/Outputs/%s_data_%s%s%04d' %(Config.rootDir, Config.mode, i[0], i[1], Config.startFile)
    else:
        store_folder='%s/Outputs/%s_data_%s%04d' %(Config.rootDir, Config.mode, i[0], Config.startFile)
    if not store_folder=="" and not os.path.exists(store_folder):
        os.mkdir(store_folder)

    if len(fileDict) > 1:
        if len(fileDict) > 28:
            mpiNodes=28
        else:
            mpiNodes=len(fileDict)
        p=Pool(mpiNodes)
        p.map(makeTFRecords, list(fileDict.keys()))
        p.close()
    else:
        targetH5File_part = list(fileDict.keys())[0]
        partList = fileDict[targetH5File_part]
        p=Pool(24)
        p.map(makeTFRecords_part, partList)
    p.close()
    p.join()
