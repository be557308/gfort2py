# SPDX-License-Identifier: GPL-2.0+

# import collections
import numpy as np
import ctypes
import six

from . import var 
from . import utils as u


# We need to store the defs here so fDT can find them during runtime 
# so we can dynamic "load" them (so we dont need to resolve them in one go
# which gets messy with recursive DT's). 
_dictDTDefs = {}


class fDT(var.fVar):
    
    def __init__(self,name=None,base_addr=-1,keys=[],key_sizes=[],key_types=[],
				key_fvars=[],dt_name=None,key_extras=[],
                pointer=False,param=False,mangled_name=None,**kwargs):
					
        self._name = name
        self._base_addr = base_addr
        self._pointer = pointer
        self._param = param
        self._mangled_name = mangled_name
        
        self._kind = None
        self._cname = 'c_void_p'
        self._ctype = getattr(ctypes,self._cname)
        self._ctype_p = ctypes.POINTER(self._ctype)
        self._pytype = dict

        if self._name is not None:
            self._up_ref()     
            
        if dt_name is not None:
            dtdf = _dictDTDefs[dt_name]
            keys = dtdf['keys']
            key_sizes = dtdf['key_sizes']
            key_fvars = dtdf['key_fvars']
            key_types = dtdf['key_types']
            key_extras = dtdf['key_extras']
    
        if len(key_sizes)==0:
            key_sizes = [ctypes.sizeof(k) for k in key_types]

        #Make offsets into a running sum
        key_sizes = [0] + key_sizes[:-1]
        offsets = np.cumsum(key_sizes)
        
        self._value={}
        for i,j,k,l,m in six.moves.zip_longest(keys,offsets,key_types,key_fvars,key_extras):
            self._value[i]={'offset':j,'ctype':k,'fvar':l,'extra':m}
                    
    def __getitem__(self,key):
        x = self._get_ref(key)
        if hasattr(x,'__ctypes_from_outparam__'):
            return x.value
        else:
            return x
        
    def __setitem__(self,key,value):
        x = self._get_ref(key)
        x.value = value

    def _get_ref(self,key):
        if key not in self._value:
            raise KeyError("""Key '"""+str(key)+"""' doesn't exist""")
            
        if self._base_addr < 0:
            raise ValueError("Must set base_addr")
            
        x_addr = self._base_addr + self._value[key]['offset']
        
        # If element is a dt we create it here
        if isinstance(self._value[key]['extra'],six.string_types):
           # x_addr = x_addr + ctypes.sizeof(ctypes.c_void_p)
            self._value[key]['extra'] = fDT(dt_name=self._value[key]['extra'],base_addr=x_addr)
        
        # Return DT if element is one
        if type(self._value[key]['extra']) is type(self):
            return self._value[key]['extra']
        else:
            return self._value[key]['ctype'].from_address(int(x_addr))
        
    def __delitem__(self,key):
        pass
        
    def __iter__(self):
        return self._value.__iter__()
        
    def __len__(self):
        return self._value.__len__()
        

    @property
    def value(self):
        return list(self._value.keys())
        
    @value.setter
    def value(self,value):
        pass

    def __dir__(self):
        return self.value

    def _ipython_key_completions_(self):
        return self.value
        
    def __contains__(self,key):
        return key in self._value
        
    def keys(self):
        return self._value.keys()
        
        

#from __future__ import print_function
#import ctypes
#import functools
#from .var import fVar
#from .cmplx import fComplex
#from .arrays import fExplicitArray, fDummyArray, fAssumedShape, fAssumedSize, fAllocatableArray
#from .strings import fStr
#from .errors import *


#_dictAllDtDescs={}


#def getEmptyDT(name):
    #class emptyDT(ctypes.Structure):
        #pass
    #emptyDT.__name__ = name
    #return emptyDT
        
#class _DTDesc(object):
    #def __init__(self,dt_def):
        #self._lib = None
        #self.dt_def = dt_def['dt_def']['arg']
        #self.dt_name = dt_def['name'].lower().replace("'","")
        
        #self.names = [i['name'].lower().replace("'","") for i in self.dt_def]
        #self.args = []
        #for i in self.dt_def:
            #self.args.append(self._init_var(i))
            
        #self.ctypes = []
        #for i in self.args:
            #try:
                #self.ctypes.append(i.ctype_def())
            #except AttributeError:
                #if hasattr(i,'dt_desc'):
                    #self.ctypes.append(i.dt_desc)
                #else:
                    #self.ctypes.append(ctypes.POINTER(getEmptyDT(self.dt_name)))
                
        #self.dt_desc = self._create_dt()
             
    #def _create_dt(self):
        #class fDerivedTypeDesc(ctypes.Structure):
            #_fields_ = list(zip(self.names,self.ctypes))
        #fDerivedTypeDesc.__name__ = self.dt_name
        #return fDerivedTypeDesc
        
    #def _init_var(self, obj):
        ## Placeholder for a dt
        #if 'dt' in obj['var']:
            #name = obj['var']['dt']['name'].lower().replace("'","")
            ##By the time we get here we should have allready filled _dictAllDtDescs
            ## with all the dt defs
            #return _dictAllDtDescs[name]
        
        #array = None
        #if 'array' in obj['var']:
            #array = obj['var']['array']
        
        #pytype = obj['var']['pytype'] 
        
        #if pytype in 'str':
            #return fStr(self._lib, obj)
        #elif pytype in 'complex':
            #return fComplex(self._lib, obj)
        #elif array is not None:
            #atype = array['atype']
            #if atype in 'explicit':
                #return fExplicitArray(self._lib, obj)
            #elif atype in 'alloc':
               #return fAllocatableArray(self._lib, obj)
            #elif atype in 'assumed_shape' or atype in 'pointer':
                #return fAssumedShape(self._lib, obj)
            #elif atype in 'assumed_size':
                #return fAssumedSize(self._lib, obj)
            #else:
                #raise ValueError("Unknown array: "+str(obj))
        #else:
           #return fVar(self._lib, obj)



#class fDerivedType(fVar):
    #def __init__(self, lib, obj):
        #self.__dict__.update(obj)
        #self._lib = lib
        #self._dt_type = self.var['dt']['name'].lower().replace("'","")

        #self._dt_desc = _dictAllDtDescs[self._dt_type]
        #self._desc = self._dt_desc.dt_desc
        #self._ctype = self._desc
        #self._ctype_desc = ctypes.POINTER(self._ctype)


        #self._args = self._dt_desc.args
        #self._nameArgs = self._dt_desc.names
        #self._typeArgs = self._dt_desc.ctypes

        #self.intent=None
        #self.pointer=None
        
        ##Store the ref to the lib object
        #try:   
            #self._ref = self._get_from_lib()
        #except NotInLib:
            #self._ref = None
        
        #self._value = {}
        
    #def get(self,copy=True):
        #res={}
        #if copy:
            #for name,i in zip(self._nameArgs,self._args):
                #x=getattr(self._ref,name)
                #res[name]=i.ctype_to_py_f(x)
        #else:
            #if hasattr(self._ref,'contents'):
                #res =self._ref.contents
            #else:
                #res = self._ref
        #return res
            
    #def set_mod(self,value):
        ## Wants a dict
        #if not all(i in self._nameArgs for i in value.keys()):
            #raise ValueError("Dict contains elements not in struct")
        
        #for name in value:
            #self.set_single(name,value[name])
            
    #def set_single(self,name,value):
        #self._setSingle(self._ref,name,value)
        
    #def _setSingle(self,v,name,value):
        #if isinstance(value,dict):
            #for i in value:
                #self._setSingle(getattr(v,name),i,value[i])
        #else:
            #setattr(v,name,value)

    #def py_to_ctype(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it
        #"""
        #self._value=self._ctype()

        ## Wants a dict
        #if isinstance(value,dict):
            #if not all(i in self._nameArgs for i in value.keys()):
                #raise ValueError("Dict contains elements not in struct")
            
            #for name in value.keys():
                #setattr(self._value,name,value[name])
        
        
        #return self._value
        
    #def py_to_ctype_f(self, value):
        #"""
        #Pass in a python value returns the ctype representation of it, 
        #suitable for a function
        
        #Second return value is anythng that needs to go at the end of the
        #arg list, like a string len
        #"""
        #r=self.py_to_ctype(value)    
            
        #return r,None

    #def ctype_to_py(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it
        #"""
        #res={}
        #for name,i in zip(self._nameArgs,self._args):
            #x=getattr(value,name)
            #res[name]=i.ctype_to_py_f(x)

        #return res
        
    #def ctype_to_py_f(self, value):
        #"""
        #Pass in a ctype value returns the python representation of it,
        #as returned by a function (may be a pointer)
        #"""
        #if hasattr(value,'contents'):
            #return self.ctype_to_py(value.contents)
        #else:
            #return self.ctype_to_py(value)

    #def ctype_def(self):
        #"""
        #The ctype type of this object
        #"""
        #return self._ctype

    #def ctype_def_func(self,pointer=False,intent=''):
        #"""
        #The ctype type of a value suitable for use as an argument of a function

        #May just call ctype_def
        
        #Second return value is anything that needs to go at the end of the
        #arg list, like a string len
        #"""
        #self.intent=intent
        #self.pointer=pointer
        #if pointer and intent is not 'na':
            #f=ctypes.POINTER(self._ctype_desc)
        #elif intent=='na':
            #f=ctypes.POINTER(self._ctype_desc)
        #else:
            #f=self._ctype_desc
            
        #return f,None
        
    #def py_to_ctype_p(self,value):
        #"""
        #The ctype representation suitable for function arguments wanting a pointer
        #"""
        #return ctypes.POINTER(self.ctype_def())(self.py_to_ctype(value))
        
    #def _pname(self):
        #return str(self.name) + " <" + str(self._dt_def['name']) + ">"

    #def __dir__(self):
        #return self._nameArgs

    #def __str__(self):
        #return self.name+" <"+self._dt_type+" dt>"
        
    #def __repr__(self):
        #return self.name+" <"+self._dt_type+" dt>"
        
    #def __getattr__(self, name): 
        #if name in self.__dict__:
            #return self.__dict__[name]

        #if '_args' in self.__dict__ and '_nameArgs' in self.__dict__:
            #if name in self._nameArgs:
                #return self.__getitem__(name)

    #def __setattr__(self, name, value):
        #if '_nameArgs' in self.__dict__:
            #if name in self._nameArgs:
                #self.set_single(name,value)
                #return
        
        #self.__dict__[name] = value
        #return    
        
    #def get_dict(self):
        #"""
        #Return a dict with the keys set suitable for this dt
        #"""
        #x={}
        #for i in self._nameArgs:
            #x[i]=0
        #return x
        
    #def __getitem__(self,name=None):
        #if name is None:
            #raise KeyError
        #if name not in self._nameArgs:
            #raise KeyError("Name not in struct")
        
        #if self._value is None or len(self._value)==0:
            #return getattr(self._ref,name)
        #else:
            #return getattr(self._value,name)
        
