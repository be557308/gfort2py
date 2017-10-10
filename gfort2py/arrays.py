from __future__ import print_function
import ctypes
from .var import fVar, fParam
import numpy as np
from .utils import *
from .fnumpy import *
from .errors import *
from .array_static import listFAllocArrays

class npFArray(np.ndarray):
    _GFC_MAX_DIMENSIONS = 7

    _GFC_DTYPE_RANK_MASK = 0x07
    _GFC_DTYPE_TYPE_SHIFT = 3
    _GFC_DTYPE_TYPE_MASK = 0x38
    _GFC_DTYPE_SIZE_SHIFT = 6

    _BT_UNKNOWN = 0
    _BT_INTEGER = _BT_UNKNOWN + 1
    _BT_LOGICAL = _BT_INTEGER + 1
    _BT_REAL = _BT_LOGICAL + 1
    _BT_COMPLEX = _BT_REAL + 1
    _BT_DERIVED = _BT_COMPLEX + 1
    _BT_CHARACTER = _BT_DERIVED + 1
    _BT_CLASS = _BT_CHARACTER + 1
    _BT_PROCEDURE = _BT_CLASS + 1
    _BT_HOLLERITH = _BT_PROCEDURE + 1
    _BT_VOID = _BT_HOLLERITH + 1
    _BT_ASSUMED = _BT_VOID + 1
    
    _index_t = ctypes.c_int64
    _size_t = ctypes.c_int64
    
    
    def __new__(cls,input_array,defined=False):
        obj = np.asarray(input_array).view(cls)
        obj._defined = defined
        return obj
        
    def __array_finalize__(self,obj):
        if obj is None:
            return
        self._ctype_desc, self._ctype = self._make_ctype_struct()
        self._defined = getattr(obj,'_defined', False)

    def _make_ctype_struct(self):
        desc = listFAllocArrays[self.ndim-1]
        ctype = desc()
        
        ctype.base_addr = self.ctypes.data
        ctype.offset = self._index_t(-1)
        ctype.dtype = self._size_t(self._get_dtype())
        
        for i in range(self.ndim):
            ctype.dims[i].stride = self._index_t(self.strides[i]//self.itemsize)
            ctype.dims[i].lbound = self._index_t(1)
            ctype.dims[i].ubound = self._index_t(self.shape[i]) 
        
        return desc, ctype
        

    def _get_dtype(self):
        ftype=self._get_ftype()
        d=self.ndim
        d=d|(ftype<<self._GFC_DTYPE_TYPE_SHIFT)
        d=d|(self.itemsize*8)<<self._GFC_DTYPE_SIZE_SHIFT
        return d

    def _get_ftype(self):
        ftype=None
        dtype = self.dtype.name
        if 'int' in dtype:
            ftype=self._BT_INTEGER
        elif 'float' in dtype:
            ftype=self._BT_REAL
        elif 'bool' in dtype:
            ftype=self._BT_LOGICAL
        elif 'str' in dtype:
            ftype=self._BT_CHARACTER
        else:
            raise ValueError("Cant match dtype, got "+dtype)
        return ftype

    @property
    def _as_parameter_(self):
        if self._defined:
            return self.ctypes.data
        else:
            return self._ctype

    @property
    def from_param(self):
        if self._defined:
            return ctypes.c_void_p
        else:
            return self._ctype_desc
            
            
            
class fArray(fVar):
    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib


        if 'array' in self.var:
            self.__dict__.update(obj['var'])

        self._ndim = int(self.array['ndims'])
        self._lib = lib
        
        defined, shape = self._array_shape()
        self._value = npFArray(np.zeros(shape),defined=defined)
        
        self._desc = self._value._desc
        self._ctype_single = getattr(ctypes,self.ctype)
        self._ctype = self._value._ctype
        self._ctype_desc = ctypes.POINTER(self._desc)
        
            
    def _array_shape(self):
        try:
            bounds = self.array['bounds']
        except KeyError:
            return False,(self._ndim)
        
        shape = []
        for i, j in zip(bounds[0::2], bounds[1::2]):
            shape.append(j - i + 1)
        return True,shape

    def _array_size(self):
        return np.product(self._make_array_shape(bounds))
            
            
    def _find_in_lib(self):
        return self._desc.in_dll(self._lib,self.mangled_name)
        
            
    def set_mod(self, value):
        """
        Set a module level variable
        """
        self._value.carray=self._find_in_lib()
        self._value._set_value(value)
        
        return 
        
    def get(self,copy=False):
        """
        Get a module level variable
        """           
        return self._value.value
        

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        self.set_func_arg(value)
        
        return self._value.ptr
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        return self.py_to_ctype(value),None

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self.ctype_to_py_f(value.contents)
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        if hasattr(value,'contents'):
            self._value.carray = value.contents
        else:
            self._value.carray = value
        
        return self._value

            
    def py_to_ctype_p(self,value):
        """
        The ctype represnation suitable for function arguments wanting a pointer
        """
        return self.py_to_ctype(value)
            

    def pytype_def(self):
        return np.array

    def ctype_def(self):
        """
        The ctype type of this object
        """
        return self._ctype_desc

    def ctype_def_func(self,pointer=False,intent=''):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """

        return self.ctype_def(),None 
            
            
              
class fDummyArray(fVar):

    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib

        print(obj)
        if 'array' in self.var:
            self.__dict__.update(obj['var'])

        self.ndim = int(self.array['ndims'])
        self._lib = lib
        
        self._desc = self._setup_desc()
        self._ctype_single = getattr(ctypes,self.ctype)
        self._ctype = self._desc
        self._ctype_desc = ctypes.POINTER(self._desc)
        self._value = self._desc(self._ctype_single,self.pytype,self._desc)
        

    def _setup_desc(self):
        return _listFAllocArrays[self.ndim]

    def set_mod(self, value):
        """
        Set a module level variable
        """
        self._value.carray=self._find_in_lib()
        self._value._set_value(value)
        
        return 
        
        
        
    def set_func_arg(self,value):
        self._value._set_value(value)

        
    def get(self,copy=False):
        """
        Get a module level variable
        """           
        return self._value.value
        

    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        self.set_func_arg(value)
        
        return self._value.ptr
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        return self.py_to_ctype(value),None

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self.ctype_to_py_f(value.contents)
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        if hasattr(value,'contents'):
            self._value.carray = value.contents
        else:
            self._value.carray = value
        
        return self._value

            
    def py_to_ctype_p(self,value):
        """
        The ctype represnation suitable for function arguments wanting a pointer
        """
        return self.py_to_ctype(value)
            

    def pytype_def(self):
        return np.array

    def ctype_def(self):
        """
        The ctype type of this object
        """
        return self._ctype_desc

    def ctype_def_func(self,pointer=False,intent=''):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """

        return self.ctype_def(),None

    def __str__(self):
        x=self.get()
        if x is None:
            return "<array>"
        else:
            return str(self.get())
        
    def __repr__(self):
        x=self.get()
        if x is None:
            return "<array>"
        else:
            return repr(self.get())

    def __getattr__(self, name): 
        if name in self.__dict__:
            return self.__dict__[name]

class fAssumedShape(fDummyArray):
    pass
    #def _get_pointer(self):
        #return self._ctype_desc.from_address(ctypes.addressof(self._value_array))
    
    
    #def set_func_arg(self,value):
        
        #super(fAssumedShape,self).set_func_arg(value)
        
        ##Fix up bounds
    
        ##From gcc source code
        ##Parsed       Lower   Upper  Returned
        ##------------------------------------
          ##:           NULL    NULL   AS_DEFERRED (*)
          ##x            1       x     AS_EXPLICIT
          ##x:           x      NULL   AS_ASSUMED_SHAPE
          ##x:y          x       y     AS_EXPLICIT
          ##x:*          x      NULL   AS_ASSUMED_SIZE
          ##*            1      NULL   AS_ASSUMED_SIZE
          
       ## for i in range(self.ndim):
            ##print(self._value_array.dims[i].lbound,self._value_array.dims[i].ubound)
            ##self._value_array.dims[i].ubound=0
            ##self._value_array.dims[i].lbound=0
            
    #def __str__(self):
        #return str(self._value_array)
        
    #def __repr__(self):
        #return repr(self._value_array)

    #def py_to_ctype(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it
        #"""
        #self.set_func_arg(value)
        #return self._value_array
        
    #def py_to_ctype_f(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it, 
        #suitable for a function
        
        #Second return value is anything that needs to go at the end of the
        #arg list, like a string len
        #"""
        #return self.py_to_ctype(value),None    
    

class fExplicitArray(fVar):

    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib
        self._pytype = np.array
        self.ctype = self.var['array']['ctype']
        
        self._ctype = self.ctype_def()
        
        if 'array' in self.var:
          self.__dict__.update(obj['var'])
        
        self.ndims = int(self.array['ndims'])
        #self._ctype_f = self.ctype_def_func()
        self._dtype=self.pytype+str(8*ctypes.sizeof(self._ctype))

        #Store the ref to the lib object
        try:   
            self._ref = self._get_from_lib()
        except NotInLib:
            self._ref = None

    def ctype_to_py(self, value):
        """
        Pass in a ctype value returns the python representation of it
        """
        return self._get_var_by_iter(value, self._array_size())
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        self._data = np.asfortranarray(value.T.astype(self._dtype))

        return self._data,None
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        self._value = np.asfortranarray(value,dtype=self._dtype)
        return self._value

    def pytype_def(self):
        return self._pytype

    def ctype_def(self):
        """
        The ctype type of this object
        """
        if '_cached_ctype' not in self.__dict__:
            self._cached_ctype = getattr(ctypes, self.ctype)
        
        return self._cached_ctype

    def ctype_def_func(self,pointer=False,intent=''):
        """
        The ctype type of a value suitable for use as an argument of a function

        May just call ctype_def
        
        Second return value is anythng that needs to go at the end of the
        arg list, like a string len
        """
        if pointer:
            raise ValueError("Cant have explicit array as a pointer")
        
        x=np.ctypeslib.ndpointer(dtype=self._dtype,ndim=self.ndims,
                                flags='F_CONTIGUOUS')
        y=None
        return x,y        
        
    def set_mod(self, value):
        """
        Set a module level variable
        """
        v = value.flatten(order='C')
        self._set_var_from_iter(self._ref, v, self._array_size())
        
    def get(self,copy=True):
        """
        Get a module level variable
        """
        s = self.ctype_to_py(self._ref)
        shape = self._make_array_shape()
        return np.reshape(s, shape)

    def _make_array_shape(self,bounds=None):
        if bounds is None:
            bounds = self.array['bounds']
        
        shape = []
        for i, j in zip(bounds[0::2], bounds[1::2]):
            shape.append(j - i + 1)
        return shape

    def _array_size(self,bounds=None):
        return np.product(self._make_array_shape(bounds))
       
    def py_to_ctype_p(self,value):
        """
        The ctype represnation suitable for function arguments wanting a pointer
        """

        raise AttributeError("Cant have explicit array as a pointer")

    
    
class fAssumedSize(fExplicitArray):
    pass
    
class fAllocatableArray(fDummyArray):
    def py_to_ctype(self, value):
        """
        Pass in a python value returns the ctype representation of it
        """
        self.set_func_arg(value)
        
        # self._value_array needs to be empty if the array is allocatable and not
        # allready allocated
        self._value_array.base_addr=ctypes.c_void_p(0)
        
        return self._value_array
        
    def py_to_ctype_f(self, value):
        """
        Pass in a python value returns the ctype representation of it, 
        suitable for a function
        
        Second return value is anything that needs to go at the end of the
        arg list, like a string len
        """
        return self.py_to_ctype(value),None   
        
    def ctype_to_py_f(self, value):
        """
        Pass in a ctype value returns the python representation of it,
        as returned by a function (may be a pointer)
        """
        shape=[]
        for i in value.dims:
            shape.append(i.ubound-i.lbound+1)
        shape=tuple(shape)
        
        p=ctypes.POINTER(self._ctype_single)
        res=ctypes.cast(value.base_addr,p)
        return np.ctypeslib.as_array(res,shape=shape)
    
class fParamArray(fParam):
    def get(self):
        """
        A parameters value is stored in the dict, as we cant access them
        from the shared lib.
        """
        return np.array(self.value, dtype=self.pytype)


