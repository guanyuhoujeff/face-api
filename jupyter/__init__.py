import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
# print('Module    sys.path  ==> ', sys.path)
add_path(ROOT_DIR)

# from controller import storeController
# from module import (
#     autoWebModule,
#     faceapiModule,
#     socketModule
# )
# print('1231')
# __all__ = [
#     "storeController",
#     "socketModule"
# ]
# print('sss')
