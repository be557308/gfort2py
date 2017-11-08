# SPDX-License-Identifier: GPL-2.0+

from __future__ import print_function
import ctypes


def find_key_val(list_dicts, key, value):
    v = value.lower()
    for idx, i in enumerate(list_dicts):
        if i[key].lower() == v:
            return idx


_lib = None
_module = ''

def loadLib(filename,mod):
    global _lib, _module
    _lib = ctypes.CDLL(filename)
    _module = '__'+str(mod)+'_MOD_'

