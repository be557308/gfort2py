from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import numpy as np
from errors import *


_lib = None

def loadLib(filename):
    global _lib
    _lib = ctypes.CDLL(filename)


def _dll(var,name):
    if name is None:
        if var.name is not None:
            name=var.name
        else:
            raise ValueError("Must set name")
            
    var.name = name

    if var._ref is None:
        try:
            var._ref = var._ctype.in_dll(_lib,name)
        except ValueError:
            raise NotInLib
    
    return var

def _in_dll(var,name=None):
    
    var = _dll(var,name)
    
    x = var._ref.value
    
    if x is None:  
        return None
    else:
        return var.__class__(value=x,**var.__dict__)
    
def _set_dll(var,value,name=None):
    
    var = _dll(var,name)
    
    var._ref.value = value
    
    return var.__class__(value=value,**var.__dict__)


def _null_ptr(var):
    return ctypes.POINTER(var._ctype)()
        

class fInt(int):
    
    #_type_ = "i"
    
    #__basicsize = ctypes.c_int.__basicsize__
    
    def __new__(cls,value=0,pointer=True,kind=4,param=False,name=None,
                *args,**kwargs):
        obj = super(fInt, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.kind = kind
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = False
        
        obj._ctype = getattr(ctypes,'c_int'+str(kind*8))
        obj._ctype_p = ctypes.POINTER(obj._ctype)
            
        obj.name = name

        obj._ref = None
        if name is not None:
            obj._ref = obj._ctype.in_dll(_lib,obj.name)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype_p(self._ctype(self.__int__()))

    @property
    def from_param(self):
        return self._ctype
        
    def in_dll(self,name=None):
        return _in_dll(self,name)

    def set_dll(self,value,name=None):
        return _set_dll(self,value,name)
        

class _fReal(float):
    def __new__(cls,value=0.0,pointer=False,kind=4,param=False,*args,**kwargs):
        obj = super(_fReal, cls).__new__(cls, value)
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
        
        obj._ctype_p = ctypes.POINTER(obj._ctype)
            
        obj.name = name

        obj._ref = None
        if name is not None:
            obj._ref = obj._ctype.in_dll(_lib,obj.name)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype_p(self._ctype(self.__float__()))

    @property
    def from_param(self):
        return self._ctype
        
    def in_dll(self,name=None):
        return _in_dll(self,name)

    def set_dll(self,value,name=None):
        return _set_dll(self,value,name)
        
        
class fSingle(_fReal):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(fSingle, cls).__new__(cls, value,pointer=pointer,kind=4)
        return obj
        
class fDouble(_fReal):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(fDouble, cls).__new__(cls, value,pointer=pointer,kind=8)
        return obj
        
        
class fQuad(np.longdouble):
    def __new__(cls,value=0.0,pointer=False,param=False,*args,**kwargs):
        obj = super(fQuad, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = False
        
        obj._ctype = ctypes.longdouble()
        
        obj._ctype_p = ctypes.POINTER(obj._ctype)
            
        obj.name = name

        obj._ref = None
        if name is not None:
            obj._ref = obj._ctype.in_dll(_lib,obj.name)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype_p(self._ctype(self.__float__()))

    @property
    def from_param(self):
        return self._ctype
        
    def in_dll(self,name=None):
        return _in_dll(self,name)

    def set_dll(self,value,name=None):
        return _set_dll(self,value,name)
        
        
class _fRealCmplx(complex):
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
        
        obj._ctype_p = ctypes.POINTER(obj._ctype)
            
        obj.name = name

        obj._ref = None
        if name is not None:
            obj._ref = obj._ctype.in_dll(_lib,obj.name)
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype_p(self._ctype(self.__float__()))

    @property
    def from_param(self):
        return self._ctype
        
    def in_dll(self,name=None):
        return _in_dll(self,name)

    def set_dll(self,value,name=None):
        return _set_dll(self,value,name)
        
        
class fSingleCmplx(_fRealCmplx):
    def __new__(cls,value=complex(0.0),pointer=False,param=False,*args,**kwargs):
        obj = super(fSingleCmplx, cls).__new__(cls, value,pointer=pointer,kind=4)
        return obj
        
class fDoubleCmplx(_fRealCmplx):
    def __new__(cls,value=complex(0.0),pointer=False,param=False,*args,**kwargs):
        obj = super(fDoubleCmplx, cls).__new__(cls, value,pointer=pointer,kind=8)
        return obj
   

class fChar(bytes):
    def __new__(cls,value=b'',pointer=False,param=False,length=-1,name=None,
                *args,**kwargs):
        obj = super(fChar, cls).__new__(cls, value)
        obj.pointer = pointer
        obj.param = param
        # True if we need extra fields in functions calls
        obj._extra = True
                
        if type(value) == 'str':
            value = value.encode()
        
        obj.length = length
        if obj.length > 0:
            if len(value) > obj.length:
                raise ValueError("String is too long")
            
        else:
            obj.length = len(value)
            
        obj._ctype = ctypes.c_char
        
        obj._ctype_p = ctypes.c_char_p
        
        obj._ref = None
        obj.name = name
        if obj.name is not None:
            obj._ref = obj._ctype.in_dll(_lib,obj.name)
        
        
        return obj
        
    @property
    def _as_parameter_(self):
        return self._ctype(self.__str__())

    @property
    def from_param(self):
        return self._ctype

    def in_dll(self,name=None):
        if name is None:
            if self.name is not None:
                name=self.name
            else:
                raise ValueError("Must set name")
                
        self.name = name
    
        if self._ref is None:
            try:
                self._ref = self._ctype_p.in_dll(_lib,name)
            except ValueError:
                raise NotInLib 

        addr = ctypes.addressof(self._ref)
        s = ctypes.string_at(addr,self.length)
        return fChar(value=s,**self.__dict__)

    def set_dll(self,value,name=None):
        if name is None:
            if self.name is not None:
                name=self.name
            else:
                raise ValueError("Must set name")
                
        self.name = name
    
        if self._ref is None:
            try:
                self._ref = self._ctype.in_dll(_lib,name)
            except ValueError:
                raise NotInLib 

        if type(value) == 'str':
            value = value.encode()

        addr = ctypes.addressof(self._ref)
        
        for j in range(min(len(value), self.length)):
            offset = addr + j * ctypes.sizeof(self._ctype)
            self._ctype.from_address(offset).value = value[j]
        
        
        return fChar(value=value,**self.__dict__)



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

