# MLGIZMO
[SPH-GCN](https://github.com/hlei-ziyan/SPH3D-GCN) based ML structure for particle-based cosmological simulation, such as [GIZMO](https://bitbucket.org/phopkins/gizmo-public/src/master/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing and development purposes.

### Prerequisites

This test needs below modules and programs well-installed.
* [TensorFlow](https://bitbucket.org/phopkins/gizmo-public/src/master/)
* [H5PY](https://github.com/h5py/h5py)
* [GIZMO](https://bitbucket.org/phopkins/gizmo-public/src/master/)
* [SPN-GCN](https://github.com/hlei-ziyan/SPH3D-GCN)

For visualization, we use the [SPLASH](http://users.monash.edu.au/~dprice/splash/).

## Running the tests

### Data Preparation

* First, you need to choose a well-tested physical problem (e.g. Evrard collapse, Blast wave, ...) in GIZMO to produce several snapshots. Here, you can check out their [documentation](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html).
* Edit the Config.py in the sub-directory (e.g. EvrardCollapse/Config.py) to specify the feature (e.g. density, internal energy, ...) you want to predict and some parameters related to the model. And for making tfrecord files for training, excecuting the hdf52tfrecord.py in the data-prepare folder.

```
python hdf52tfrecord.py [config file]
```


### Break down into end to end tests

* Excute ```python train.py/evaluate.py [feature]``` to train/evaluate the model.

This test shows how well the SPH-GCN can do to capture physical significances embedded in the simulation results from the state-of-the-art code. To convert the model prediction to the hdf5 file, please execute the ```matToHDF5.py``` and one can use the SPLASH code to visualize the results, like
![alt text](https://github.com/seanthchao/MLGIZMO/blob/master/Image/residualMap.png)
