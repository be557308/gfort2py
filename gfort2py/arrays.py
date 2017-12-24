# SPDX-License-Identifier: GPL-2.0+

from __future__ import print_function
import ctypes
import collections
import numpy as np
import sys

from .errors import *
from . import dt
from . import var
from . import utils as u
from . import fnumpy


if sys.byteorder is 'little':
    _byte_order=">"
else:
    _byte_order="<"

class fArray(var.fVar):
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
    
    _BT_TYPESPEC = {_BT_UNKNOWN:'v',_BT_INTEGER:'i',_BT_LOGICAL:'b',
                    _BT_REAL:'g',_BT_COMPLEX:'c',_BT_DERIVED:'v',
                    _BT_CHARACTER:'v',_BT_CLASS:'v',_BT_PROCEDURE:'v',
                    _BT_HOLLERITH:'v',_BT_VOID:'v',_BT_ASSUMED:'v'}
                    
    _PY_TO_BT = {'int':_BT_INTEGER,'float':_BT_REAL,'bool':_BT_LOGICAL,
                'str':_BT_CHARACTER,'bytes':_BT_CHARACTER}

    _index_t = ctypes.c_int64
    _size_t = ctypes.c_int64
    
    _array_keys = ['array_addr','offset','dtype']
    _array_types = [ctypes.c_void_p,_size_t,_index_t]
    
    _dims_keys = ['stride','lbound','ubound']
    _dims_types = [_index_t,_index_t,_index_t]
    _bounds = collections.namedtuple('_bounds',_dims_keys)
    
    def __init__(self,pointer=True,kind=-1,name=None,param=False,
                    base_addr=-1,cname=None,pytype=None,ctype=None,
                    shape=None,ndim=None,explicit=True,mangled_name=None,
                    **kwargs):
    
        if shape is None and ndim is None:
            raise ValueError("Must set either shape or ndim")
            
    
        self._farray = None
        self._value = None  
        self._shape = None
        self._ndim = None 
        self._pyalloc = False
        self._name = None
        self._value = None
        self._explicit = explicit
        self._mangled_name = mangled_name
                            
        if shape is not None:
            print(shape)
            if not hasattr(shape,'__iter__'):
                raise ValueError("Shape must be an iterable")
            self._shape = tuple(shape)
            self._ndims = len(shape)
            self._falloc = True
                
        if ndim is not None:
            if ndim < 0:
                raise ValueError("ndim must be > 0")
            self._ndim = ndim
            self._falloc = True

        if base_addr > 0:
            self._base_addr = base_addr
        else:
            self._base_addr = -1
    
        if self._explicit and self._shape is None:
            raise ValueError("Explicit arrays must have thier shape set")
    
    
        self._pointer = pointer
        self._kind = kind
        self._name = name
        self._param = param
 
        self._cname = cname
        self._ctype = ctypes.c_void_p
        self._ctype_p = ctypes.c_void_p
                
        #python type of indivdual element
        self._pytype_s = pytype
        #ctype of a single element
        self._ctype_s = ctype
        
        self._init_farray()
        
        if self.name is not None:
            self._up_ref()
    
    def _init_farray(self):
        #Create storage area for fortran array:
        self._farray = dt.fDT(keys=self._create_keys(self._ndim),
                              key_types=self._create_types(self._ndim),
                              base_addr=-1)
        
        self._farray.__array_interface__ = {
                'shape': None,
                'typestr': None,
                'data': None,
                'strides': None,
                'version': 3, # Only use the latest version
                }
        
    
    @property
    def value(self):
        self._up_ref()
        
        if self._base_addr < 0:
            raise ValueError("Must set either the name or base_addr of array")
        
        if self._ref.value is None:
            raise ArrayNotAllocated("Array hasn't been allocated yet")
        
        
        if self._explicit:
            #These are defined only by a pointer to the first element
            BT = self._PY_TO_BT[str(self._pytype_s.__name__)]
            size = ctypes.sizeof(self._ctype_s)

            strides = tuple(int(np.product(self._shape[0:i])) for i,v in enumerate(self._shape))
            self._farray.__array_interface__['data'] = (self._base_addr,False)
            
        else:            
            self._ndim, BT, size = self._split_dtype(self._farray['dtype'])
            self._shape = self._get_shape()
            strides = self._get_strides()
            self._farray.__array_interface__['data'] = (self._ref.value,False)
            
        self._farray.__array_interface__['shape'] = self._shape
        self._farray.__array_interface__['typestr'] = self._BT_to_typestr(BT)+str(size)
        if self._ndim > 1:
            self._farray.__array_interface__['strides'] = tuple(i*size for i in strides)
        else:
            self._farray.__array_interface__['strides'] = None
        
        print(self._farray.__array_interface__)
        self._value = np.array(self._farray,copy=False)
        print("*",self._value)
        fnumpy.remove_ownership(self._value)
        self._falloc = True
        self._pyalloc = False
        
        return self._value
       
    @value.setter
    def value(self,value):
        # TODO: Add logic to deallocate the old array
        if self._param:
            raise ValueError("Cant alter a parameter")
        
        self._value = value
        fnumpy.remove_ownership(self._value)
        self._shape = value.shape
        self._ndim = value.ndim
        self._pyalloc = True
        
        #self._up_ref()
        self._farray.__array_interface__ = self._value.__array_interface__
        
        # This misses the first 8 bytes
        #self._base_addr = self._farray.__array_interface__['data'][0]
        # This sets only the first 8 bytes
        #ctypes.memmove(self._base_addr,self._value.ctypes.data, ctypes.sizeof(ctypes.c_void_p))
       
       
        if not self._explicit:
            self._farray['dtype'] = self._create_dtype(ndim=self._ndim, 
                                                   itemsize=ctypes.sizeof(self._ctype_s),
                                                   ftype=str(self._pytype_s)
                                                   )
            self._set_bounds()
        print(self._farray.__array_interface__)
        
        
        
    def _get_bounds(self):
        bounds = []
        for i in range(self._ndim):
            bounds.append(self._bounds(
                          self._farray['stride_'+str(i)],
                          self._farray['lbound_'+str(i)],
                          self._farray['ubound_'+str(i)],
                          ))
        return bounds
        
    def _get_shape(self):
        bounds = self._get_bounds()
        shape = []
        for i in bounds:
            shape.append(i.ubound-i.lbound+1)
        
        return tuple(shape) #__array_interface__ needs a tuple not a list
    
    def _get_strides(self):
        bounds = self._get_bounds()
        strides = []
        for i in bounds:
            strides.append(i.stride)
        
        return tuple(strides)
        
    def _set_bounds(self):
        bounds = []
        for i in range(self._ndim):
            self._farray['lbound_'+str(i)] = 1
            self._farray['ubound_'+str(i)] = self._value.shape[i]
            self._farray['stride_'+str(i)] = int(np.product(self._value.shape[0:i]))



    def _create_dtype(self,ndim,itemsize,ftype):
        ftype=self._get_BT(ftype)
        d=ndim
        d=d|(ftype<<self._GFC_DTYPE_TYPE_SHIFT)
        d=d|int(itemsize)<<self._GFC_DTYPE_SIZE_SHIFT
        return d
    
    def _get_BT(self,ftype):
        if 'int' in ftype:
            BT=self._BT_INTEGER
        elif 'float' in ftype:
            BT=self._BT_REAL
        elif 'bool' in ftype:
            BT=self._BT_LOGICAL
        elif 'str' in ftype or 'bytes' in ftype:
            BT=self._BT_CHARACTER
        else:
            raise ValueError("Cant match dtype, got "+ftype)
        return BT
        
    def _BT_to_typestr(self,BT):
        try:
            res = self._BT_TYPESPEC[BT]
        except KeyError:
            raise BadFortranArray("Bad BT value "+str(BT))
            
        return _byte_order+res
    

    def _split_dtype(self,dtype):
        itemsize = dtype >> self._GFC_DTYPE_SIZE_SHIFT
        BT = (dtype >> self._GFC_DTYPE_TYPE_SHIFT ) & (self._GFC_DTYPE_RANK_MASK)
        ndim = dtype & self._GFC_DTYPE_RANK_MASK
        
        return ndim,BT,int(itemsize)

    def _array_names(self,names,ndim):
        res = []
        for j in range(ndim):
            for i in names:
                res.append(str(i)+"_"+str(j))
        return res
    
    
    def _create_keys(self,ndim):
        return self._array_keys+self._array_names(self._dims_keys,ndim)
    
    
    def _create_types(self,ndim):
        return self._array_types+self._dims_types*ndim
        
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
        else:
            self._ref = None
            self._base_addr = -1
        
    def _up_ref(self):
        if self._base_addr < 0:
            try:
                self._ref = ctypes.c_void_p.in_dll(u._lib,self._mod_name)
                self._base_addr = ctypes.addressof(self._ref)
            except ValueError:
                raise NotInLib 
                
        
        self._farray._base_addr = self._base_addr
        self._farray['array_addr'] = self._ref.value
        
#####################################################################

#class npFArray(np.ndarray):
    #_GFC_MAX_DIMENSIONS = 7

    #_GFC_DTYPE_RANK_MASK = 0x07
    #_GFC_DTYPE_TYPE_SHIFT = 3
    #_GFC_DTYPE_TYPE_MASK = 0x38
    #_GFC_DTYPE_SIZE_SHIFT = 6

    #_BT_UNKNOWN = 0
    #_BT_INTEGER = _BT_UNKNOWN + 1
    #_BT_LOGICAL = _BT_INTEGER + 1
    #_BT_REAL = _BT_LOGICAL + 1
    #_BT_COMPLEX = _BT_REAL + 1
    #_BT_DERIVED = _BT_COMPLEX + 1
    #_BT_CHARACTER = _BT_DERIVED + 1
    #_BT_CLASS = _BT_CHARACTER + 1
    #_BT_PROCEDURE = _BT_CLASS + 1
    #_BT_HOLLERITH = _BT_PROCEDURE + 1
    #_BT_VOID = _BT_HOLLERITH + 1
    #_BT_ASSUMED = _BT_VOID + 1
    
    #_index_t = ctypes.c_int64
    #_size_t = ctypes.c_int64
    
    
    #def __new__(cls,input_array,defined=False):
        #obj = np.asarray(input_array).view(cls)
        #obj._defined = defined
        #return obj
        
    #def __array_finalize__(self,obj):
        #if obj is None:
            #return
        #self._ctype_desc, self._ctype = self._make_ctype_struct()
        #self._defined = getattr(obj,'_defined', False)

    #def _make_ctype_struct(self):
        #desc = listFAllocArrays[self.ndim-1]
        #ctype = desc()
        
        #ctype.base_addr = self.ctypes.data
        #ctype.offset = self._index_t(-1)
        #ctype.dtype = self._size_t(self._get_dtype())
        
        #for i in range(self.ndim):
            #ctype.dims[i].stride = self._index_t(self.strides[i]//self.itemsize)
            #ctype.dims[i].lbound = self._index_t(1)
            #ctype.dims[i].ubound = self._index_t(self.shape[i]) 
        
        #return desc, ctype
        

    #def _get_dtype(self):
        #ftype=self._get_ftype()
        #d=self.ndim
        #d=d|(ftype<<self._GFC_DTYPE_TYPE_SHIFT)
        #d=d|(self.itemsize*8)<<self._GFC_DTYPE_SIZE_SHIFT
        #return d

    #def _get_ftype(self):
        #ftype=None
        #dtype = self.dtype.name
        #if 'int' in dtype:
            #ftype=self._BT_INTEGER
        #elif 'float' in dtype:
            #ftype=self._BT_REAL
        #elif 'bool' in dtype:
            #ftype=self._BT_LOGICAL
        #elif 'str' in dtype:
            #ftype=self._BT_CHARACTER
        #else:
            #raise ValueError("Cant match dtype, got "+dtype)
        #return ftype

    #@property
    #def _as_parameter_(self):
        #if self._defined:
            #return self.ctypes.data
        #else:
            #return self._ctype

    #@property
    #def from_param(self):
        #if self._defined:
            #return ctypes.c_void_p
        #else:
            #return self._ctype_desc
            
            
            
#class fArray(fVar):
    #def __init__(self, lib, obj):
        #self.__dict__.update(obj)
        #self._lib = lib


        #if 'array' in self.var:
            #self.__dict__.update(obj['var'])

        #self._ndim = int(self.array['ndims'])
        #self._lib = lib
        
        #defined, shape = self._array_shape()
        #self._value = npFArray(np.zeros(shape),defined=defined)
        
        #self._desc = self._value._desc
        #self._ctype_single = getattr(ctypes,self.ctype)
        #self._ctype = self._value._ctype
        #self._ctype_desc = ctypes.POINTER(self._desc)
        
            
    #def _array_shape(self):
        #try:
            #bounds = self.array['bounds']
        #except KeyError:
            #return False,(self._ndim)
        
        #shape = []
        #for i, j in zip(bounds[0::2], bounds[1::2]):
            #shape.append(j - i + 1)
        #return True,shape

    #def _array_size(self):
        #return np.product(self._make_array_shape(bounds))
            
            
    #def _find_in_lib(self):
        #return self._desc.in_dll(self._lib,self.mangled_name)
        
            
    #def set_mod(self, value):
        #"""
        #Set a module level variable
        #"""
        #self._value.carray=self._find_in_lib()
        #self._value._set_value(value)
        
        #return 
        
    #def get(self,copy=False):
        #"""
        #Get a module level variable
        #"""           
        #return self._value.value
        

    #def py_to_ctype(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it
        #"""
        #self.set_func_arg(value)
        
        #return self._value.ptr
        
    #def py_to_ctype_f(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it, 
        #suitable for a function
        
        #Second return value is anything that needs to go at the end of the
        #arg list, like a string len
        #"""
        #return self.py_to_ctype(value),None

    #def ctype_to_py(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it
        #"""
        #return self.ctype_to_py_f(value.contents)
        
    #def ctype_to_py_f(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it,
        #as returned by a function (may be a pointer)
        #"""
        #if hasattr(value,'contents'):
            #self._value.carray = value.contents
        #else:
            #self._value.carray = value
        
        #return self._value

            
    #def py_to_ctype_p(self,value):
        #"""
        #The ctype represnation suitable for function arguments wanting a pointer
        #"""
        #return self.py_to_ctype(value)
            

    #def pytype_def(self):
        #return np.array

    #def ctype_def(self):
        #"""
        #The ctype type of this object
        #"""
        #return self._ctype_desc

    #def ctype_def_func(self,pointer=False,intent=''):
        #"""
        #The ctype type of a value suitable for use as an argument of a function

        #May just call ctype_def
        
        #Second return value is anythng that needs to go at the end of the
        #arg list, like a string len
        #"""

        #return self.ctype_def(),None 
            
            
              
#class fDummyArray(fVar):

    #def __init__(self, lib, obj):
        #self.__dict__.update(obj)
        #self._lib = lib

        #print(obj)
        #if 'array' in self.var:
            #self.__dict__.update(obj['var'])

        #self.ndim = int(self.array['ndims'])
        #self._lib = lib
        
        #self._desc = self._setup_desc()
        #self._ctype_single = getattr(ctypes,self.ctype)
        #self._ctype = self._desc
        #self._ctype_desc = ctypes.POINTER(self._desc)
        #self._value = self._desc(self._ctype_single,self.pytype,self._desc)
        

    #def _setup_desc(self):
        #return _listFAllocArrays[self.ndim]

    #def set_mod(self, value):
        #"""
        #Set a module level variable
        #"""
        #self._value.carray=self._find_in_lib()
        #self._value._set_value(value)
        
        #return 
        
        
        
    #def set_func_arg(self,value):
        #self._value._set_value(value)

        
    #def get(self,copy=False):
        #"""
        #Get a module level variable
        #"""           
        #return self._value.value
        

    #def py_to_ctype(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it
        #"""
        #self.set_func_arg(value)
        
        #return self._value.ptr
        
    #def py_to_ctype_f(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it, 
        #suitable for a function
        
        #Second return value is anything that needs to go at the end of the
        #arg list, like a string len
        #"""
        #return self.py_to_ctype(value),None

    #def ctype_to_py(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it
        #"""
        #return self.ctype_to_py_f(value.contents)
        
    #def ctype_to_py_f(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it,
        #as returned by a function (may be a pointer)
        #"""
        #if hasattr(value,'contents'):
            #self._value.carray = value.contents
        #else:
            #self._value.carray = value
        
        #return self._value

            
    #def py_to_ctype_p(self,value):
        #"""
        #The ctype represnation suitable for function arguments wanting a pointer
        #"""
        #return self.py_to_ctype(value)
            

    #def pytype_def(self):
        #return np.array

    #def ctype_def(self):
        #"""
        #The ctype type of this object
        #"""
        #return self._ctype_desc

    #def ctype_def_func(self,pointer=False,intent=''):
        #"""
        #The ctype type of a value suitable for use as an argument of a function

        #May just call ctype_def
        
        #Second return value is anythng that needs to go at the end of the
        #arg list, like a string len
        #"""

        #return self.ctype_def(),None

    #def __str__(self):
        #x=self.get()
        #if x is None:
            #return "<array>"
        #else:
            #return str(self.get())
        
    #def __repr__(self):
        #x=self.get()
        #if x is None:
            #return "<array>"
        #else:
            #return repr(self.get())

    #def __getattr__(self, name): 
        #if name in self.__dict__:
            #return self.__dict__[name]

#class fAssumedShape(fDummyArray):
    #pass
    ##def _get_pointer(self):
        ##return self._ctype_desc.from_address(ctypes.addressof(self._value_array))
    
    
    ##def set_func_arg(self,value):
        
        ##super(fAssumedShape,self).set_func_arg(value)
        
        ###Fix up bounds
    
        ###From gcc source code
        ###Parsed       Lower   Upper  Returned
        ###------------------------------------
          ###:           NULL    NULL   AS_DEFERRED (*)
          ###x            1       x     AS_EXPLICIT
          ###x:           x      NULL   AS_ASSUMED_SHAPE
          ###x:y          x       y     AS_EXPLICIT
          ###x:*          x      NULL   AS_ASSUMED_SIZE
          ###*            1      NULL   AS_ASSUMED_SIZE
          
       ### for i in range(self.ndim):
            ###print(self._value_array.dims[i].lbound,self._value_array.dims[i].ubound)
            ###self._value_array.dims[i].ubound=0
            ###self._value_array.dims[i].lbound=0
            
    ##def __str__(self):
        ##return str(self._value_array)
        
    ##def __repr__(self):
        ##return repr(self._value_array)

    ##def py_to_ctype(self, value):
        ##"""
        ##Pass in a python value returns the ctype representation of it
        ##"""
        ##self.set_func_arg(value)
        ##return self._value_array
        
    ##def py_to_ctype_f(self, value):
        ##"""
        ##Pass in a python value returns the ctype representation of it, 
        ##suitable for a function
        
        ##Second return value is anything that needs to go at the end of the
        ##arg list, like a string len
        ##"""
        ##return self.py_to_ctype(value),None    
    

#class fExplicitArray(fVar):

    #def __init__(self, lib, obj):
        #self.__dict__.update(obj)
        #self._lib = lib
        #self._pytype = np.array
        #self.ctype = self.var['array']['ctype']
        
        #self._ctype = self.ctype_def()
        
        #if 'array' in self.var:
          #self.__dict__.update(obj['var'])
        
        #self.ndims = int(self.array['ndims'])
        ##self._ctype_f = self.ctype_def_func()
        #self._dtype=self.pytype+str(8*ctypes.sizeof(self._ctype))

        ##Store the ref to the lib object
        #try:   
            #self._ref = self._get_from_lib()
        #except NotInLib:
            #self._ref = None

    #def ctype_to_py(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it
        #"""
        #return self._get_var_by_iter(value, self._array_size())
        
    #def py_to_ctype_f(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it, 
        #suitable for a function
        
        #Second return value is anything that needs to go at the end of the
        #arg list, like a string len
        #"""
        #self._data = np.asfortranarray(value.T.astype(self._dtype))

        #return self._data,None
        
    #def ctype_to_py_f(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it,
        #as returned by a function (may be a pointer)
        #"""
        #self._value = np.asfortranarray(value,dtype=self._dtype)
        #return self._value

    #def pytype_def(self):
        #return self._pytype

    #def ctype_def(self):
        #"""
        #The ctype type of this object
        #"""
        #if '_cached_ctype' not in self.__dict__:
            #self._cached_ctype = getattr(ctypes, self.ctype)
        
        #return self._cached_ctype

    #def ctype_def_func(self,pointer=False,intent=''):
        #"""
        #The ctype type of a value suitable for use as an argument of a function

        #May just call ctype_def
        
        #Second return value is anythng that needs to go at the end of the
        #arg list, like a string len
        #"""
        #if pointer:
            #raise ValueError("Cant have explicit array as a pointer")
        
        #x=np.ctypeslib.ndpointer(dtype=self._dtype,ndim=self.ndims,
                                #flags='F_CONTIGUOUS')
        #y=None
        #return x,y        
        
    #def set_mod(self, value):
        #"""
        #Set a module level variable
        #"""
        #v = value.flatten(order='C')
        #self._set_var_from_iter(self._ref, v, self._array_size())
        
    #def get(self,copy=True):
        #"""
        #Get a module level variable
        #"""
        #s = self.ctype_to_py(self._ref)
        #shape = self._make_array_shape()
        #return np.reshape(s, shape)

    #def _make_array_shape(self,bounds=None):
        #if bounds is None:
            #bounds = self.array['bounds']
        
        #shape = []
        #for i, j in zip(bounds[0::2], bounds[1::2]):
            #shape.append(j - i + 1)
        #return shape

    #def _array_size(self,bounds=None):
        #return np.product(self._make_array_shape(bounds))
       
    #def py_to_ctype_p(self,value):
        #"""
        #The ctype represnation suitable for function arguments wanting a pointer
        #"""

        #raise AttributeError("Cant have explicit array as a pointer")

    
    
#class fAssumedSize(fExplicitArray):
    #pass
    
#class fAllocatableArray(fDummyArray):
    #def py_to_ctype(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it
        #"""
        #self.set_func_arg(value)
        
        ## self._value_array needs to be empty if the array is allocatable and not
        ## allready allocated
        #self._value_array.base_addr=ctypes.c_void_p(0)
        
        #return self._value_array
        
    #def py_to_ctype_f(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it, 
        #suitable for a function
        
        #Second return value is anything that needs to go at the end of the
        #arg list, like a string len
        #"""
        #return self.py_to_ctype(value),None   
        
    #def ctype_to_py_f(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it,
        #as returned by a function (may be a pointer)
        #"""
        #shape=[]
        #for i in value.dims:
            #shape.append(i.ubound-i.lbound+1)
        #shape=tuple(shape)
        
        #p=ctypes.POINTER(self._ctype_single)
        #res=ctypes.cast(value.base_addr,p)
        #return np.ctypeslib.as_array(res,shape=shape)
    
#class fParamArray(fParam):
    #def get(self):
        #"""
        #A parameters value is stored in the dict, as we cant access them
        #from the shared lib.
        #"""
        #return np.array(self.value, dtype=self.pytype)


