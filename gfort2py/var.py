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
        self._param = param
        
        self._cname = cname
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
        if not self._param:
            self._up_ref()
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
        if not self._param:
            if self._base_addr > 0:
                x = self._ctype.from_address(self._base_addr)
            elif self._ref is not None:
                x = self._ref
            else:
                raise ValueError("Value not mapped to fortran")        
            
            x.value = self._pytype(x)
            self._value = x.value
        else:
            raise AttributeError("Can not alter a parameter")

        
    def _set_from_buffer(self):
        for i in range(self._length):
            offset = self.base_addr + i * ctypes.sizeof(self._ctype)
            self._ctype.from_address(offset).value = self._value[i]
        
    def _get_from_buffer(self):
        self._value = []
        for i in range(self._length):
            offset = self.base_addr + i * ctypes.sizeof(self._ctype)
            self._value[i] = self._ctype.from_address(offset).value
        
        
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
    
    def __len__(self):
        y = self._get()
        return y.__len__() 


class fInt(fVar):

    def __init__(self,value=0,pointer=True,kind=4,param=False,name=None,
                base_addr=-1,*args,**kwargs):
                    
        cname = 'c_int'+str(kind*8)
        pytpe = int
        super(fInt, self).__init__(value=value,pointer=pointer,kind=kind,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
    
        
class fSingle(fVar):
   def __init__(self,value=0.0,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_float'
        pytpe = np.longdouble
        super(fQuad, self).__init__(value=value,pointer=pointer,kind=4,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
        
class fDouble(fVar):
   def __init__(self,value=0.0,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_double'
        pytpe = np.longdouble
        super(fQuad, self).__init__(value=value,pointer=pointer,kind=8,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
        
class fQuad(fVar):
   def __init__(self,value=np.longdouble(0.0),pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_longdouble'
        pytpe = np.longdouble
        super(fQuad, self).__init__(value=value,pointer=pointer,kind=16,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)

        
class fLogical(fVar):
   def __init__(self,value=True,pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_bool'
        pytpe = bool
        super(fLogical, self).__init__(value=value,pointer=pointer,kind=kind,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)

        
    
        
        
class fSingleCmplx(fVar):
   def __init__(self,value=cmplx(0.0),pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_float'*2
        pytpe = complex
        super(fSingleCmplx, self).__init__(value=value,pointer=pointer,kind=2*4,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
        
        self._length = 2
        
    @property
    def _as_parameter_(self):
        x = self._get()
        y = ctypes._ctype(x.real,x.imag)
        return self._ctype_p(y)
        
    @property
    def from_param(self):
        return self._ctype

    def _get(self,new=True):
        if self._base_addr > 0:
            x = self._get_from_buffer(self._base_addr)
        else:
            x = [self._value.real,self._value.imag]
            
        self._value = self._pytype(x[0],x[1])

        return self._value
        
    def _set(self,value):
        if self._base_addr < 0:
            raise ValueError("Value not mapped to fortran")        
        
        self._value = self._pytype(value)
        
        self._set_from_buffer()
        
class fDoubleCmplx(fVar):
   def __init__(self,value=cmplx(0.0),pointer=True,param=False,name=None,
                base_addr=-1,*args,**kwargs):

        cname = 'c_double'*2
        pytpe = complex
        super(fDoubleCmplx, self).__init__(value=value,pointer=pointer,kind=2*8,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
        
        self._length = 2
        
    @property
    def _as_parameter_(self):
        x = self._get()
        y = ctypes._ctype(x.real,x.imag)
        return self._ctype_p(y)
        
    @property
    def from_param(self):
        return self._ctype

    def _get(self,new=True):
        if self._base_addr > 0:
            x = self._get_from_buffer(self._base_addr)
        else:
            x = [self._value.real,self._value.imag]
            
        self._value = self._pytype(x[0],x[1])

        return self._value
        
    def _set(self,value):
        if not self._param:
            if self._base_addr < 0:
                raise ValueError("Value not mapped to fortran")   
        else:
            raise AttributeError("Can not alter a parameter")
        
        self._value = self._pytype(value)
        
        self._set_from_buffer()
   

class fChar(fVar):
    def __init_(self,value=b'',pointer=True,param=False,name=None,length=-1
                base_addr=-1,*args,**kwargs):
        cname = 'c_char_p'
        pytpe = bytes
        self._length = length
        super(fChar, self).__init__(value=value,pointer=pointer,kind=2*length,
                                    param=param,name=name,base_addr=base_addr,
                                    cname=cname,pytype=pytype,
                                    *args,**kwargs)
                                    

    @property
    def _as_parameter_(self):
        return self._ctype_p(self._ctype(self._get()))
        
    @property
    def from_param(self):
        return self._ctype

    def _get(self,new=True):
        if self._base_addr > 0:
            x = ctypes.string_at(self._base_addr,self.length)
        else:
            x = self._value
            
        try:
            self._value = self._pytype(x)
        except TypeError:
            self._value = self._pytype(x.encode())

        return self._value
        
    def _set(self,value):
        if not self._param:
            if self._base_addr < 0:
                raise ValueError("Value not mapped to fortran")   
        else:
            raise AttributeError("Can not alter a parameter")        
        
        try:
            self._value = self._pytype(value)
        except TypeError:
            self._value = self._pytype(value.encode())
        
        self._length = len(self._value)
        
        self._set_from_buffer()

    def _extra_ctype(self):
        return ctypes.c_int
        
    def _extra_val(self,x):
        return len(x)
