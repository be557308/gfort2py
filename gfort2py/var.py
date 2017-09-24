from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import numpy as np
from .errors import *

class fInt(int):
    def __new__(cls,value=0,pointer=False,kind=4,param=False,*args,**kwargs):
        obj = super(fInt, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.kind = kind
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = False
        
        obj._ctype = getattr(ctypes,'c_int'+kind*8)
        if pointer:
            obj._ctype = ctypes.POINTER(obj._ctype)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype(self.__int__())

    @property
    def from_param(self):
        return self._ctype
        
    def _null_ptr(self):
        return ctypes.POINTER(obj._ctype)()

class _fReal(float):
    def __new__(cls,value=0.0,pointer=False,kind=4,param=False,*args,**kwargs):
        obj = super(fReal, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.kind = kind
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = False
        
        if kind==4:
            obj._ctype = ctypes.c_float
        elif kind==8:
            obj._ctype = ctypes.c_double
        else:
            raise ValueError("Kind must be 4 or 8")
        
        if pointer:
            obj._ctype = ctypes.POINTER(obj._ctype)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype(self.__float__())

    @property
    def from_param(self):
        return self._ctype
        
        
class fSingle(float):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(_fReal, cls).__new__(cls, value,pointer=pointer,kind=4)
        return obj
        
class fDouble(float):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(_fReal, cls).__new__(cls, value,pointer=pointer,kind=8)
        return obj
        
        
class fQuad(np.longdouble):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(fQuad, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = False
        
        obj._ctype = ctypes.longdouble()
        
        if pointer:
            obj._ctype = ctypes.POINTER(obj._ctype)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return obj._ctype(self.view())

    @property
    def from_param(self):
        return obj._ctype
        
        
class _fRealCmplx(float):
    def __new__(cls,value=complex(0.0),pointer=False,kind=4,param=False,*args,**kwargs):
        obj = super(fReal, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.kind = kind
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = False
        
        if kind==4:
            obj._ctype = ctypes.c_float*2
        elif kind==8:
            obj._ctype = ctypes.c_double*2
        else:
            raise ValueError("Kind must be 4 or 8")
        
        if pointer:
            obj._ctype = ctypes.POINTER(obj._ctype)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return [self.real,self.imag]

    @property
    def from_param(self):
        return self._ctype
        
        
class fSingle(float):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(_fReal, cls).__new__(cls, value,pointer=pointer,kind=4)
        return obj
        
class fDouble(float):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(_fReal, cls).__new__(cls, value,pointer=pointer,kind=8)
        return obj
        
        
class fQuad(np.longdouble):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(fQuad, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = False
        
        obj._ctype = ctypes.longdouble()
        
        if pointer:
            obj._ctype = ctypes.POINTER(obj._ctype)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype(self.view())

    @property
    def from_param(self):
        return self._ctype


class fStr(bytes):
    def __new__(cls,value=b'',pointer=False,param=False,length=-1,*args,**kwargs):
        obj = super(fReal, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.kind = kind
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = True
        
        obj.length = length
        
        if type(value) == 'str':
            value = value.encode()
            
        if obj.length > 0 and len(value) > obj.length:
            raise ValueError("String is too long")
            
        obj._ctype = ctypes.c_char
        
        if pointer:
            obj._ctype = ctypes.c_char_p
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype(self.__float__())

    @property
    def from_param(self):
        return self._ctype

    @property
    def _extra_ctype(self):
        return ctypes.c_int
        
    @property
    def _extra_val(self):
        return len(self)


    def _convert(self,value):
        #handle array of bytes back to a string?
        pass


    def _get_from_lib(self):
        if 'mangled_name' in self.__dict__ and '_lib' in self.__dict__:
            try:
                return self._ctype.in_dll(self._lib, self.mangled_name)
            except ValueError:
                raise NotInLib
        raise NotInLib
        

    #maybe use ctypes.string_at(addr,size)?
    def _get_var_by_iter(self, value, size=-1):
        """ Gets a variable where we have to iterate to get multiple elements"""
        base_address = ctypes.addressof(value)
        return self._get_var_from_address(base_address, size=size)

    def _get_var_from_address(self, ctype_address, size=-1):
        out = []
        i = 0
        sof = ctypes.sizeof(self._ctype)
        while True:
            if i == size:
                break
            x = self._ctype.from_address(ctype_address + i * sof)
            if x.value == b'\x00':
                break
            else:
                out.append(x.value)
            i = i + 1
        return out

    def _set_var_from_iter(self, res, value, size=99999):
        base_address = ctypes.addressof(res)
        self._set_var_from_address(base_address, value, size)

    def _set_var_from_address(self, ctype_address, value, size=99999):
        for j in range(min(len(value), size)):
            offset = ctype_address + j * ctypes.sizeof(self._ctype)
            self._ctype.from_address(offset).value = value[j]

