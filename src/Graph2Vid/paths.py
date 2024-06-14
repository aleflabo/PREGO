import sys
import os

# root project and weights folder
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH = os.path.join(PROJECT_PATH, "weights")
CT_PATH = os.path.join(PROJECT_PATH, "Datasets", 'crosstask')
S3D_PATH = "/user/n.dvornik/Git/S3D_HowTo100M/"  # change this
