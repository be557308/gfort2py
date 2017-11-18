# SPDX-License-Identifier: GPL-2.0+

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
    def __init__(self,pointer=True,kind=-1,name=None,
                    base_addr=-1,cname=None,pytype=None,param=False,
                    *args,**kwargs):
        
        self._value = None
        self._pointer = pointer
        self._kind = kind
        self._name = name
        self._base_addr = base_addr
        self._param = param
        
        self._cname = cname
        self._ctype = getattr(ctypes,cname)
        self._ctype_p = ctypes.POINTER(self._ctype)
        
        self._ref = None
        if self.name is not None:
            self._up_ref()
            
        self._pytype = pytype
        
    @property
    def _mod_name(self):
        res = ''
        if self.name is not None:
            return u._module + self.name
        return res 
        
    @property    
    def name(self):
        return self._name
        
    @name.setter
    def name(self,name):
        self._name = str(name)
        if not self._param:
            self._up_ref()
            if self._ref.value is None:
                raise ValueError("Bad name")
            self._base_addr = ctypes.addressof(self._ref)
        else:
            self._ref = None
            self._base_addr = -1

    def set_addr(self,addr):
        if not self._param:
            self._base_addr = addr
            self._ref = None
        else:
            self._ref = None
            self._base_addr = -1
        
    def _up_ref(self):
        try:
            self._ref = self._ctype.in_dll(u._lib,self._mod_name)
            self._base_addr = ctypes.addressof(self._ref)
        except ValueError:
            raise NotInLib 
        
    @property
    def _as_parameter_(self):
        return self._ctype_p(self._ctype(self._get()))
        
    @property
    def from_param(self):
        return self._ctype

    @property
    def value(self):
        if self._base_addr > 0:
            x = self._ctype.from_address(self._base_addr).value
        elif self._ref is not None:
            x = self._ref.value
        else:
            x = self._value
            
        self._value = self._pytype(x)

        return self._value
      
    @value.setter
    def value(self,value):
        if not self._param:
            if self._base_addr > 0:
                x = self._ctype.from_address(self._base_addr)
            elif self._ref is not None:
                x = self._ref
            else:
                raise ValueError("Value not mapped to fortran")        
            x.value = self._pytype(value)
            self._value = x.value
        else:
            raise AttributeError("Can not alter a parameter")

        
    def _set_from_buffer(self):
        for i in range(self._length):
            offset = self._base_addr + i * ctypes.sizeof(self._ctype)
            self._ctype.from_address(offset).value = self._value[i]
        
    def _get_from_buffer(self):
        self._value = []
        for i in range(self._length):
            offset = self._base_addr + i * ctypes.sizeof(self._ctype)
            self._value[i] = self._ctype.from_address(offset).value
        
        
    # self.value should allways be a python type so we overload and let
    # the python types __X__ functions act    
    
    def __add__(self,x):
        return self.value.__add__(x)
        
    def __sub__(self,x):
        return self.value.__sub__(x)
    
    def __mul__(self,x):
        return self.value.__mul__(x)
        
    def __divmod__(self,x):
        return self.value.__divmod__(x)
        
    def __truediv__(self,x):
        return self.value.__truediv__(x)
        
    def __floordiv__(self,x):
        return self.value.__floordiv__(x)
        
    def __matmul_(self,x):
        return self.value.__matmul__(x) 

    def __pow__(self,x,modulo=None):
        return self.value.__pow__(x,modulo)
        
    def __lshift__(self,x):
        return self.value.__lshift__(x)
        
    def __rshift__(self,x):
        return self.value.__rshift__(x)
        
    def __and__(self,x):
        return self.value.__and__(x)
        
    def __xor__(self,x):
        return self.value.__xor__(x)        

    def __or__(self,x):
        return self.value.__or__(x)
        
    def __radd__(self,x):
        return self.value.__radd__(x)
        
    def __rsub__(self,x):
        return self.value.__rsub__(x)
    
    def __rmul__(self,x):
        return self.value.__rmul__(x)
        
    def __rdivmod__(self,x):
        return self.value.__rdivmod__(x)
        
    def __rtruediv__(self,x):
        return self.value.__rtruediv__(x)
        
    def __rfloordiv__(self,x):
        return self.value.__rfloordiv__(x)
        
    def __rmatmul_(self,x):
        return self.value.__rmatmul__(x)

    def __rpow__(self,x):
        return self.value.__rpow__(x)
        
    def __rlshift__(self,x):
        return self.value.__rlshift__(x)
        
    def __rrshift__(self,x):
        return self.value.__rrshift__(x)
        
    def __rand__(self,x):
        return self.value.__rand__(x)
        
    def __rxor__(self,x):
        return self.value.__rxor__(x)        

    def __ror__(self,x):
        return self.value.__ror__(x)
   
    def __iadd__(self,x):
        self.value = self.value.__add__(x)
        return self.value
        
    def __isub__(self,x):
        self.value = self.value.__sub__(x)
        return self.value
    
    def __imul__(self,x):
        self.value = self.value.__mul__(x)
        return self.value
        
    def __itruediv__(self,x):
        self.value = self.value.__truediv__(x)
        return self.value
        
    def __ifloordiv__(self,x):
        self.value = self.value.__floordiv__(x)
        return self.value
        
    def __imatmul_(self,x):
        self.value = self.value.__matmul__(x)
        return self.value

    def __ipow__(self,x,modulo=None):
        self.value = self.value.__pow__(x,modulp)
        return self.value
        
    def __ilshift__(self,x):
        self.value = self.value.__lshift__(x)
        return self.value
        
    def __irshift__(self,x):
        self.value = self.value.__rshift__(x)
        return self.value
        
    def __iand__(self,x):
        self.value = self.value.__and__(x)
        return self.value
        
    def __ixor__(self,x):
        self.value = self.value.__xor__(x)
        return self.value        

    def __ior__(self,x):
        self.value = self.value.__or__(x)
        return self.value
        
    def __str__(self):
        return self.value.__str__()
        
    def __repr__(self):
        return self.value.__repr__()
   
    def __bytes__(self):
        return self.value.__bytes__()
 
    def __format__(self):
        return self.value.__format__()
        
    def __lt__(self,x):
        return self.value.__lt__()
        
    def __le__(self,x):
        return self.value.__le__()
        
    def __gt__(self,x):
        return self.value.__gt__()
        
    def __ge__(self,x):
        return self.value.__ge__()
        
    def __eq__(self,x):
        return self.value.__eq__()
        
    def __ne__(self,x):
        return self.value.__ne__()

    def __bool__(self):
        return self.value.__bool__()

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()
        
    def __abs__(self):
        return self.value.__abs__()
        
    def __invert__(self):
        return self.value.__invert__()
        
    def __complex__(self):
        return self.value.__complex__()
        
    def __int__(self):
        return self.value.__int__()
        
    def __float__(self):
        return self.value.__float__()
        
    def __round__(self):
        return self.value.__round__()
    
    def __len__(self):
        return self.value.__len__()
        
    def __hash__(self):
        raise NotImplemented


class fInt(fVar):

    def __init__(self,pointer=True,kind=4,param=False,name=None,
                base_addr=-1,*args,**kwargs):
                    
        cname = 'c_int'+str(kind*8)
        pytype = int
        super(fInt, self).__init__(pointer=pointer,kind=kind,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
                                    
        self._value = self._pytype(0.0)
    
        
class fSingle(fVar):
   def __init__(self,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_float'
        pytype = float
        super(fSingle, self).__init__(pointer=pointer,kind=4,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
                                    
        self._value = self._pytype(0.0)
        
class fDouble(fVar):
   def __init__(self,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_double'
        pytype = float
        super(fDouble, self).__init__(pointer=pointer,kind=8,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
                                    
        self._value = self._pytype(0.0)
        
class fQuad(fVar):
   def __init__(self,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_longdouble'
        pytype = np.longdouble
        super(fQuad, self).__init__(pointer=pointer,kind=16,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
                                    
        self._value = self._pytype(0.0)

        
class fLogical(fVar):
   def __init__(self,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_bool'
        pytype = bool
        super(fLogical, self).__init__(pointer=pointer,kind=kind,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
                                    
        self._value = self._pytype(0)
        
    
        
        
class fSingleCmplx(fVar):
    def __init__(self,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_float'*2
        pytype = complex
        super(fSingleCmplx, self).__init__(pointer=pointer,kind=2*4,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
        
        self._length = 2
        self._value = self._pytype(0.0)

    @property
    def _as_parameter_(self):
        x = self._get()
        y = ctypes._ctype(x.real,x.imag)
        return self._ctype_p(y)
    
    @property
    def value(self,new=True):
        if self._base_addr > 0:
            x = self._get_from_buffer(self._base_addr)
        else:
            x = [self._value.real,self._value.imag]
            
        self._value = self._pytype(x[0],x[1])

        return self._value
      
    @value.setter
    def value(self,value):
        if self._base_addr < 0:
            raise ValueError("Value not mapped to fortran")        
        
        self._value = self._pytype(value)
        
        self._set_from_buffer()
        
class fDoubleCmplx(fVar):
    def __init__(self,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_double'*2
        pytype = complex
        super(fDoubleCmplx, self).__init__(pointer=pointer,kind=2*8,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
        
        self._length = 2
        self._value = self._pytype(0.0)

    @property
    def _as_parameter_(self):
        x = self._get()
        y = ctypes._ctype(x.real,x.imag)
        return self._ctype_p(y)

    @property
    def value(self,new=True):
        if self._base_addr > 0:
            x = self._get_from_buffer(self._base_addr)
        else:
            x = [self._value.real,self._value.imag]
            
        self._value = self._pytype(x[0],x[1])

        return self._value
        
    @value.setter
    def value(self,value):
        if not self._param:
            if self._base_addr < 0:
                raise ValueError("Value not mapped to fortran")   
        else:
            raise AttributeError("Can not alter a parameter")
        
        self._value = self._pytype(value)
        
        self._set_from_buffer()
   

class fChar(fVar):
    def __init__(self,pointer=True,param=False,name=None,length=-1,
                base_addr=-1,*args,**kwargs):
                    
        cname = 'c_char'
        pytype = bytes
        if length < 0:
            raise ValueError("Must set max length of the character string")
        else:
            self._length = length
        super(fChar, self).__init__(pointer=pointer,kind=length,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
        
        self._value = self._pytype(b'')
        
    @property
    def value(self):
        if self._base_addr > 0:
            x = ctypes.string_at(self._base_addr,self._length)
        else:
            x = self._value
            
        try:
            self._value = self._pytype(x)
        except TypeError:
            self._value = self._pytype(x.encode())

        return self._value
        
    @value.setter
    def value(self,value):
        if not self._param:
            if self._base_addr < 0:
                raise ValueError("Value not mapped to fortran")   
        else:
            raise AttributeError("Can not alter a parameter")        
        
        #Truncate and possibblt pad string to fit inside the max length of the character
        value = value[0:self._length].ljust(self._length)
        
        try:
            self._value = self._pytype(value)
        except TypeError:
            self._value = self._pytype(value.encode())
            
        self._set_from_buffer()

    def _extra_ctype(self):
        return ctypes.c_int
        
    def _extra_val(self,x):
        return len(x)
        
