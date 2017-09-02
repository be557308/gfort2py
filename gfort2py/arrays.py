from __future__ import print_function
import ctypes
from .var import fVar, fParam
import numpy as np
from .utils import *
from .fnumpy import *
from .errors import *

_index_t = ctypes.c_int64
_size_t = ctypes.c_int64


class _fArrayBase(ctypes.Structure):
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
    
    def __init__(self,ctype,pytype,desc,carray=None):
        self._value = None
        self.ctype = ctype # ctype of a single element int32 float64 etc
        self.carray = carray #This is the ctypes object either from a getattr(self.lib,self.mangled_name) or what is returned from a function
        self.pytype = pytype
        self.npdtype=self.pytype+str(8*ctypes.sizeof(self.ctype))
        self.desc = desc
        
        self.set_ptr()
            
        super(ctypes.Structure, self).__init__()
   
    def _get_pointer(self):
        if self.carray is not None:
            return self.desc.from_address(ctypes.addressof(self.carray))
        else:
            raise TypeError
            
    def set_ptr(self):
        try:
            self.ptr = self._get_pointer()
        except TypeError:
            self.ptr = None
    
    @property
    def value(self):
        self.set_ptr()
        if not self.isallocated():
            return np.zeros(1)
            
        base_addr = self.ptr.base_addr
        offset = self.ptr.offset
        dtype = self.ptr.dtype
        
        dims=[]
        shape=[]
        for i in range(self.ndims):
            dims.append({})
            dims[i]['stride'] = self.ptr.dims[i].stride
            dims[i]['lbound'] = self.ptr.dims[i].lbound
            dims[i]['ubound'] = self.ptr.dims[i].ubound
            
        for i in range(self.ndims):
            shape.append(dims[i]['ubound']-dims[i]['lbound']+1)
            
        shape=tuple(shape)
        size = np.product(shape)
        
        ptr = ctypes.cast(base_addr,ctypes.POINTER(self.ctype))
        self._value = np.ctypeslib.as_array(ptr,shape=shape)
    
        return self._value
        
    def __setattr__(self,name,value):
        if name is 'value':
            self._set_value(value)
        else:
            self.__dict__[name]=value
        
    def _set_value(self,value):
        self.set_ptr()
        # Grab a copy of the data
        self._value = value.astype(self.npdtype,copy=True)
        if not self.ndims == self._value.ndim:
            raise ValueError("Array size mismatch "+str(self.ndims)+' '+str(self._value.ndim))  
            
        if self._value.ndim > self._GFC_MAX_DIMENSIONS:
            raise ValueError("Array too big")

        # Let fortran own the memoray
        remove_ownership(self._value)

        if self.ptr is not None:
            self.ptr.base_addr = self._value.ctypes.get_data()
            self.ptr.offset = _size_t(-1)
            
            self.ptr.dtype = self._get_dtype()
            
            for i in range(self.ndims):
                self.ptr.dims[i].stride = _index_t(self._value.strides[i]//ctypes.sizeof(self.ctype))
                self.ptr.dims[i].lbound = _index_t(1)
                self.ptr.dims[i].ubound = _index_t(self._value.shape[i])    
    
    def isallocated(self): 
        self.set_ptr()   
        if self.ptr is not None:  
            if self.ptr.base_addr:
                #Base addr is NULL if deallocated
                return True
        return False

            
    def _get_dtype(self):
        ftype=self._get_ftype()
        d=self.ndims
        d=d|(ftype<<self._GFC_DTYPE_TYPE_SHIFT)
        d=d|(ctypes.sizeof(self.ctype)<<self._GFC_DTYPE_SIZE_SHIFT)
        return d

    def _get_ftype(self):
        ftype=None
        dtype=str(self.ctype)
        if 'c_int' in dtype:
            ftype=self._BT_INTEGER
        elif 'c_double' in dtype or 'c_real' in dtype or 'c_float' in dtype:
            ftype=self._BT_REAL
        elif 'c_bool' in dtype:
            ftype=self._BT_LOGICAL
        elif 'c_char' in dtype:
            ftype=self._BT_CHARACTER
        else:
            raise ValueError("Cant match dtype, got "+dtype)
        return ftype
        
    def __del__(self):
        if '_value' in self.__dict__:
            #Problem occurs as both fortran and numpy are pointing to same memory address
            #Thus if fortran deallocates the array numpy will try to free the pointer
            #when del is called casuing a double free error
            
            #By calling remove_ownership we tell numpy it dosn't own the data
            #thus is shouldn't call free(ptr).
            if not self.isallocated():
                remove_ownership(self._value)

            #Leaks if fortran doesn't dealloc the array
            


# Pre generate alloc array descriptors
class _bounds(ctypes.Structure):
    _fields_=[("stride",_index_t),
              ("lbound",_index_t),
              ("ubound",_index_t)]

class _fAllocArray1D(_fArrayBase):
    ndims=1
    _fields_=[('base_addr',ctypes.c_void_p), 
              ('offset',_size_t), 
              ('dtype',_index_t),
              ('dims',_bounds*ndims)
              ]

class _fAllocArray2D(_fArrayBase):
    ndims=2
    _fields_=[('base_addr',ctypes.c_void_p), 
              ('offset',_size_t), 
              ('dtype',_index_t),
              ('dims',_bounds*ndims)
              ]
              
class _fAllocArray3D(_fArrayBase):
    ndims=3
    _fields_=[('base_addr',ctypes.c_void_p), 
              ('offset',_size_t), 
              ('dtype',_index_t),
              ('dims',_bounds*ndims)
              ]
              
class _fAllocArray4D(_fArrayBase):
    ndims=4
    _fields_=[('base_addr',ctypes.c_void_p), 
              ('offset',_size_t), 
              ('dtype',_index_t),
              ('dims',_bounds*ndims)
              ]
class _fAllocArray5D(_fArrayBase):
    ndims=5
    _fields_=[('base_addr',ctypes.c_void_p), 
              ('offset',_size_t), 
              ('dtype',_index_t),
              ('dims',_bounds*ndims)
              ]
              
class _fAllocArray6D(_fArrayBase):
    ndims=6
    _fields_=[('base_addr',ctypes.c_void_p), 
              ('offset',_size_t), 
              ('dtype',_index_t),
              ('dims',_bounds*ndims)
              ]
              
class _fAllocArray7D(_fArrayBase):
    ndims=7
    _fields_=[('base_addr',ctypes.c_void_p), 
              ('offset',_size_t), 
              ('dtype',_index_t),
              ('dims',_bounds*ndims)
              ]

# None is in there so we can do 1 based indexing
_listFAllocArrays=[None,_fAllocArray1D,_fAllocArray2D,_fAllocArray3D,
                    _fAllocArray4D,_fAllocArray5D,_fAllocArray6D,
                    _fAllocArray7D] 

              
class fDummyArray(fVar):

    def __init__(self, lib, obj):
        self.__dict__.update(obj)
        self._lib = lib

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
        
    def _find_in_lib(self):
        return self._desc.in_dll(self._lib,self.mangled_name)
        
        
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


