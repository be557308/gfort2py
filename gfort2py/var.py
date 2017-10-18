from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import numpy as np
from errors import *
import utils as u


class fVar(object):
    def __init__(self,value=None,pointer=True,kind=-1,name=None,
                    base_addr=-1,cname=None,pytype=None,*args,**kwargs):
        if value is None:
            raise ValueError("Must set a value")
            
        self._value = value

        self._pointer = pointer
        self._kind = kind
        self._name = name
        self._base_addr = base_addr
        
        self._ctype = getattr(ctypes,cname)
        self._ctype_p = ctypes.POINTER(self._ctype)
        
        self._ref = None
        if self._name is not None:
            self._mod_name = u.module + self.name
            self._up_ref()
            
        self._pytype = pytype
        
        
    def set_name(self,name):
        self._name = str(name)
        self._mod_name = u.module + self.name
        self._up_ref()
        self._base_addr = -1

    def set_addr(self,addr):
        self._base_addr = addr
        self._ref = None
        
    def _up_ref(self):
        try:
            self._ref = self._ctype.in_dll(_lib,self._mod_name)
        except ValueError:
            raise NotInLib 
        
    @property
    def _as_parameter_(self):
        return self._ctype_p(self._ctype(self._get()))
        
    @property
    def from_param(self):
        return self._ctype

    def _get(self,new=True):
        if self._base_addr > 0:
            x = self._ctype.from_address(self._base_addr).value
        elif self._ref is not None:
            x = self._ref.value
        else:
            x = self._value
            
        self._value = self._pytype(x)

        return self._value
        
    def _set(self,value):
        if self._base_addr > 0:
            x = self._ctype.from_address(self._base_addr)
        elif self._ref is not None:
            x = self._ref
        else:
            raise ValueError("Value not mapped to fortran")        
        
        x.value = self._pytype(x)
        self._value = x.value
        
    # _get() should allways return a python type so we overload and let
    # the python types __X__ functions act    
    
    def __add__(self,x):
        y = self._get()
        y = y.__add__(x)
        return y
        
    def __sub__(self,x):
        y = self._get()
        y = y.__sub__(x)
        return y
    
    def __mul__(self,x):
        y = self._get()
        y = y.__mul__(x)
        return y
        
    def __divmod__(self,x):
        y = self._get()
        y = y.__divmod__(x)
        return y
        
    def __truediv__(self,x):
        y = self._get()
        y = y.__truediv__(x)
        return y
        
    def __floordiv__(self,x):
        y = self._get()
        y = y.__floordiv__(x)
        return y
        
    def __matmul_(self,x):
        y = self._get()
        y = y.__matmul__(x)
        return y 

    def __pow__(self,x,modulo=None)
        y = self._get()
        y = y.__pow__(x,modulo)
        return y 
        
    def __lshift__(self,x)
        y = self._get()
        y = y.__lshift__(x)
        return y 
        
    def __rshift__(self,x)
        y = self._get()
        y = y.__rshift__(x)
        return y 
        
    def __and__(self,x)
        y = self._get()
        y = y.__and__(x)
        return y 
        
    def __xor__(self,x)
        y = self._get()
        y = y.__xor__(x)
        return y         

    def __or__(self,x)
        y = self._get()
        y = y.__or__(x)
        return y 
        
   
    def __radd__(self,x):
        y = self._get()
        y = y.__radd__(x)
        return y
        
    def __rsub__(self,x):
        y = self._get()
        y = y.__rsub__(x)
        return y
    
    def __rmul__(self,x):
        y = self._get()
        y = y.__rmul__(x)
        return y
        
    def __rdivmod__(self,x):
        y = self._get()
        y = y.__rdivmod__(x)
        return y
        
    def __rtruediv__(self,x):
        y = self._get()
        y = y.__rtruediv__(x)
        return y
        
    def __rfloordiv__(self,x):
        y = self._get()
        y = y.__rfloordiv__(x)
        return y
        
    def __rmatmul_(self,x):
        y = self._get()
        y = y.__rmatmul__(x)
        return y 

    def __rpow__(self,x)
        y = self._get()
        y = y.__rpow__(x)
        return y 
        
    def __rlshift__(self,x)
        y = self._get()
        y = y.__rlshift__(x)
        return y 
        
    def __rrshift__(self,x)
        y = self._get()
        y = y.__rrshift__(x)
        return y 
        
    def __rand__(self,x)
        y = self._get()
        y = y.__rand__(x)
        return y 
        
    def __rxor__(self,x)
        y = self._get()
        y = y.__rxor__(x)
        return y         

    def __ror__(self,x)
        y = self._get()
        y = y.__ror__(x)
        return y 
   
    def __iadd__(self,x):
        y = self._get()
        y = y.__add__(x)
        self._set(y)
        return y
        
    def __isub__(self,x):
        y = self._get()
        y = y.__sub__(x)
        self._set(y)
        return y
    
    def __imul__(self,x):
        y = self._get()
        y = y.__mul__(x)
        self._set(y)
        return y
        
    def __itruediv__(self,x):
        y = self._get()
        y = y.__truediv__(x)
        self._set(y)
        return y
        
    def __ifloordiv__(self,x):
        y = self._get()
        y = y.__floordiv__(x)
        self._set(y)
        return y
        
    def __imatmul_(self,x):
        y = self._get()
        y = y.__matmul__(x)
        self._set(y)
        return y 

    def __ipow__(self,x,module=None)
        y = self._get()
        y = y.__pow__(x,modulo)
        self._set(y)
        return y 
        
    def __ilshift__(self,x)
        y = self._get()
        y = y.__lshift__(x)
        self._set(y)
        return y 
        
    def __irshift__(self,x)
        y = self._get()
        y = y.__rshift__(x)
        self._set(y)
        return y 
        
    def __iand__(self,x)
        y = self._get()
        y = y.__and__(x)
        self._set(y)
        return y 
        
    def __ixor__(self,x)
        y = self._get()
        y = y.__xor__(x)
        self._set(y)
        return y         

    def __ior__(self,x)
        y = self._get()
        y = y.__or__(x)
        self._set(y)
        return y 
        
    def __str__(self):
        y = self._get()
        return y.__str__()
        
    def __repr__(self):
        y = self._get()
        return y.__repr__()
   
    def __bytes__(self):
        y = self._get()
        return y.__bytes__()
 
    def __format__(self):
        y = self._get()
        return y.__format__()
        
    def __lt__(self,x):
        y = self._get()
        return y.__lt__(x)
        
    def __le__(self,x):
        y = self._get()
        return y.__le__(x)
        
    def __gt__(self,x):
        y = self._get()
        return y.__gt__(x)
        
    def __ge__(self,x):
        y = self._get()
        return y.__ge__(x)
        
    def __eq__(self,x):
        y = self._get()
        return y.__eq__(x)
        
    def __ne__(self,x):
        y = self._get()
        return y.__ne__(x)

    def __bool__(self):
        y = self._get()
        return y.__bool__() 

    def __neg__(self):
        y = self._get()
        return y.__neg__() 

    def __pos__(self):
        y = self._get()
        return y.__pos__() 
        
    def __abs__(self):
        y = self._get()
        return y.__abs__() 
        
    def __invert__(self):
        y = self._get()
        return y.__invert__() 
        
    def __complex__(self):
        y = self._get()
        return y.__complex__() 
        
    def __int__(self):
        y = self._get()
        return y.__int__() 
        
    def __float__(self):
        y = self._get()
        return y.__float__() 
        
    def __round__(self):
        y = self._get()
        return y.__round__() 
    



class fInt(fVar):

    def __init__(self,value=0,pointer=True,kind=4,param=False,name=None,
                base_addr=-1,*args,**kwargs):
                    
        self._cname = 'c_int'+str(kind*8)
        self._pytpe = int
        super(fInt, self).__init__(value=value,pointer=pointer,kind=kind,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=self.cname,pytyp=self.pytype,
                                    *args,**kwargs)
    

        


class _fReal(fVar):
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

        
class fLogical(fVar):
    pass
        
        
class _fRealCmplx(fVar):
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
   

class fChar(fVar):
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

    def in_dll(self,name=None):

        addr = ctypes.addressof(self._ref)
        s = ctypes.string_at(addr,self.length)
        return fChar(value=s,**self.__dict__)

    def set_dll(self,value,name=None):

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
        return len(self.__str__())


    def _convert(self,value):
        #handle array of bytes back to a string?
        pass


    #maybe use ctypes.string_at(addr,size)?
    #def _get_var_by_iter(self, value, size=-1):
        #""" Gets a variable where we have to iterate to get multiple elements"""
        #base_address = ctypes.addressof(value)
        #return self._get_var_from_address(base_address, size=size)

    #def _get_var_from_address(self, ctype_address, size=-1):
        #out = []
        #i = 0
        #sof = ctypes.sizeof(self._ctype)
        #while True:
            #if i == size:
                #break
            #x = self._ctype.from_address(ctype_address + i * sof)
            #if x.value == b'\x00':
                #break
            #else:
                #out.append(x.value)
            #i = i + 1
        #return out

    #def _set_var_from_iter(self, res, value, size=99999):
        #base_address = ctypes.addressof(res)
        #self._set_var_from_address(base_address, value, size)

    #def _set_var_from_address(self, ctype_address, value, size=99999):
        #for j in range(min(len(value), size)):
            #offset = ctype_address + j * ctypes.sizeof(self._ctype)
            #self._ctype.from_address(offset).value = value[j]

