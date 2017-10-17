from __future__ import print_function
import ctypes


def find_key_val(list_dicts, key, value):
    v = value.lower()
    for idx, i in enumerate(list_dicts):
        if i[key].lower() == v:
            return idx


_lib = None

def loadLib(filename):
    global _lib
    _lib = ctypes.CDLL(filename)
