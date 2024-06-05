import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

THIS_DIR = os.path.dirname(__file__)
JUPYTER_DIR = os.path.dirname(THIS_DIR)
ROOT_DIR = os.path.dirname(JUPYTER_DIR)
# print('Module    sys.path  ==> ', sys.path)
add_path(ROOT_DIR)

