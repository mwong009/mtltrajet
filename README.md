# Machine Learning based Multiple Discrete Continuous (ML-MDC) Behavioural Choice Modelling

## File description

```logit.py``` script to estimate the logit type model

```mixedlogit.py``` script to estimate the mixed logit type model

```network.py``` core functionality

```optimizers.py``` gradient descent optimizers

```compile.py``` script for generating the dataset

```main.py``` main script for estimating the CRBM model

```utility.py``` various mathematical functions and tools


### Discrete variable numbering values

purpose = {
    0: [null], 1: work, 2: education, 3: home, 4: errand, 5: health,
    6: leisure, 7: shopping, 8: meals, 9: meeting
}

mode = {
    1: CYCLING, 2: DRIVING, 3: DRIVING + TRANSIT,
    4: TRANSIT, 5: WALK, 6: OTHER
}

## Dataset

The [dataset](https://github.com/LiTrans/ML-MDC/blob/master/datatable_sm.csv) is a small sample from the Mtl Trajet 2016 dataset

## Getting started

run ```python3 main.py``` script to start the estimation to reproduce the results of the CRBM model

### Prerequisites

Python 3.5+ (with pip3), Numpy, Pandas, Theano

### Installation

A ```requirements.txt``` file is provided to install the required library packages through pip

- clone or download the git project repository, and in the project folder run the following to install the reuqirements

#### Ubuntu (Unix)

The following system packages are required to be installed

```
apt-get install python3 python3-dev pip3
python3 --version
>>> Python 3.X.X
```

Install requirements with pip with --user option

```
cd project-root-folder/
pip3 install --user -r requirements.txt
```

The above command also installs the latest Theano from github.com/Theano/Theano

#### Windows

Two options:
- Install Python directly ([instructions](https://wiki.python.org/moin/BeginnersGuide/Download))
- By Anaconda ([instructions](https://www.anaconda.com/distribution/))

verify Python is installed correctly:

Open *cmd* and run:

```
C:\>python
> Python 3.X.X. ...
```

Install project requirements

```
cd project-root-folder/
pip install -r requirements.txt
```

## Versioning

0.1 Initial version

## Authors

Melvin Wong ([Github](https://github.com/mwong009))

## Licence

This project is licensed under the MIT - see [LICENSE](https://github.com/LiTrans/ML-MDC/blob/master/LICENSE) for details
