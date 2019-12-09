# ECEN 649 Project (Fall 2019)
Implementation of Viola-Jones Algorithm. The project is about identifying the faces and non-faces and predicting them for the test data using Adaboost.

## Getting Started
The environment and the packages used in the code are given below. In addition to this, the document also includes the configuration of the computer used. Since this was run as a single threaded code, the details about the GPU are not necessary. For ease of use, the project is divided into 4 different codes. Each of the codes run in exactly the same way, but serve different purposes for the sake of the problem statement.

### Prerequisites
A virtual environment was created using Anaconda3. The code was then run on Jupyter Notebook in that virtual environment. Following commands were used to create the virtual environment

```
# Create a sample environment for the project
>> conda create --name ecen649 python=3.6
```

The hardware used to run this code was an Intel Core-i7 4750HQ processor running at a speed of 2 GHz and overclocked to 3.9 GHz. In addition to this, the computer had 16 GB Memory out of which 15.4 GB was used while running the code.

## Running the code
The code requires the following packages to run

```
%matplotlib inline

import os
import tarfile
import shutil
import hashlib
import glob
import random
from datetime import datetime
from typing import *


import requests
from joblib import Parallel, delayed
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from sklearn.metrics import *
import seaborn as sns

import matplotlib.pyplot as plt
```

These packages are used at different places in the code. More details are given in the comments of the code.

## Deployment
There is no direct deployment of the code. The code has been written in the Jupyter Notebook and the output of the code is available in the notebook itself. Along with this, the report gives a more detailed explanation of the outputs obtained through the code.
