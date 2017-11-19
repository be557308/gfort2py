# SPDX-License-Identifier: GPL-2.0+

from __future__ import print_function
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

import ctypes
import os
import six
import select
import collections
import itertools

#from .var import fVar
#from .cmplx import fComplex
#from .arrays import fExplicitArray, fDummyArray, fAssumedShape, fAssumedSize, fAllocatableArray
#from .strings import fStr
#from .types import fDerivedType
import utils as u
from errors import *


_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None

class captureStdOut():
    def read_pipe(self,pipe_out):
        def more_data():
            r, _, _ = select.select([pipe_out], [], [], 0)
            return bool(r)
        out = b''
        while more_data():
            out += os.read(pipe_out, 1024)
        return out.decode()
    
    def __enter__(self):
        if _TEST_FLAG:
            self.pipe_out, self.pipe_in = os.pipe()
            self.stdout = os.dup(1)
            os.dup2(self.pipe_in, 1)
    
    def __exit__(self,*args,**kwargs):
        if _TEST_FLAG:
            os.dup2(self.stdout, 1)
            print(self.read_pipe(self.pipe_out))
            os.close(self.pipe_in)
            os.close(self.pipe_out)
            os.close(self.stdout)

_args = collections.namedtuple('_args',('fvar','intent','opt'))

class fFunc(object):
    def __init__(self,name,arg_return=None,arg_names=[],arg_types=[],
                arg_opts=[],arg_intents=[]):
        self._name = name
        self._arg_return = arg_return

        self._args=collections.OrderedDict()
                
        for n,v,o,i in itertools.zip_longest(arg_names,arg_types,arg_opts,arg_intents,fillvalue=None):            
            self._args[n]=_args(v,i,o)

        try:
            self._call = getattr(u._lib, self._mod_name)
        except AttributeError:
            print("Skipping "+self._mod_name)
            return
            
        if self._arg_return is not None:
            self._call.restype = arg_return._ctype

    @property
    def _mod_name(self):
        res = ''
        if self.name is not None:
            return u._module + self.name
        return res 
        
    @property    
    def name(self):
        return self._name


    def __call__(self,**kwargs):
        
        # Argument processing
        args=[]
        for name,info in self._args.items():
            if name not in kwargs:
                raise AttributeError("Missing argument "+str(name))
            value=kwargs[name]
            
            if value is None:
                #Optional arguments:
                if not info.opt is None:
                    # Check if optional is in arg list
                    raise ValueError("Passed as optional, non optional argument "+str(name))
                args.append(ctypes.c_void_p(None))
            elif (info.intent is not 'in') or info.fvar._pointer:
                #Intents inout, out or unknown
                args.append(info.fvar._ctype_p(info.fvar._ctype(value)))
            else:
                info.fvar.value=value
                args.append(info.fvar._ctype(value))

        # Capture stdout messages
        with captureStdOut() as cs:  
            # Doesn't like empty lists      
            if len(args):
                res = self._call(*args)
            else:
                res = self._call()

        # Deal with intent inout,out,unknown arguments
        args_out={}
        for name,value in zip(self._args.keys(),args):
            info = self._args[name]
                
            if (info.intent is not 'in'):
                #Intents inout, out or unknown
                args_out[name]=info.fvar
                # Should proberbly do something with pointers for arrays
                args_out[name].value = value.contents.value
        
        
        if self._arg_return is not None:
            args_out[self.name] = self._arg_return
            args_out[self.name].value = res

        return args_out


#class fFunc(fVar):

    #def __init__(self, lib, obj):
        #self.__dict__.update(obj)
        #self._lib = lib
        #self._sub = self.proc['sub']
        #try:
            #self._call = getattr(self._lib, self.mangled_name)
        #except AttributeError:
            #print("Skipping "+self.mangled_name)
            #return
            
        #self._set_return()
        #self._set_arg_ctypes()
        #self.save_args=False
        #self.args_out = None

    #def _set_arg_ctypes(self):
        #self._arg_ctypes = []
        #self._arg_vars = []
        
        #tmp=[]
        #if len(self.proc['arg_nums'])>0:
            #for i in self.arg:
                #self._arg_vars.append(self._init_var(i))
                
                #if 'pointer' in i['var']:
                    #pointer=True
                #else:
                    #pointer=False
                #x,y=self._arg_vars[-1].ctype_def_func(pointer=pointer,intent=i['var']['intent'])
                #self._arg_ctypes.append(x)
                #if y is not None:
                    #tmp.append(y)
                    
            #self._call.argtypes = self._arg_ctypes+tmp

    #def _init_var(self, obj):
        #array = None
        #if 'array' in obj['var']:
            #array = obj['var']['array']
        
        #if obj['var']['pytype'] == 'str':
            #x = fStr(self._lib, obj)
        #elif obj['var']['pytype'] == 'complex':
            #x = fComplex(self._lib, obj)
        #elif 'dt' in obj['var']:
            #x = fDerivedType(self._lib, obj)
        #elif array is not None:
            ##print(self.name,array)
            #if array['atype'] == 'explicit':
                #x = fExplicitArray(self._lib, obj)
            #elif array['atype'] == 'alloc':
                #x = fAllocatableArray(self._lib, obj)
            #elif array['atype'] == 'assumed_shape' or array['atype'] == 'pointer':
                #x = fAssumedShape(self._lib, obj)
            #elif array['atype'] == 'assumed_size':
                #x = fAssumedSize(self._lib, obj)
            #else:
                #raise ValueError("Unknown array: "+str(obj))
        #else:
            #x = fVar(self._lib, obj)

        #x._func_arg=True

        #return x

    #def _set_return(self):
        #if not self._sub:
            #self._restype = self.ctype_def()
            #self._call.restype = self._restype
            
    #def _args_to_ctypes(self,args):
        #tmp = []
        #args_in = []
        #for vout, vin, fctype, a in six.moves.zip_longest(self._arg_vars, args, self._arg_ctypes, self.arg):
            #if 'optional' in a['var'] and vin is None:
                ##Missing optional arguments 
                #args_in.append(None)            
            #else:
                #x,y=vout.py_to_ctype_f(vin)
                #if 'pointer' in a['var']:
                    #args_in.append(vout.py_to_ctype_p(vin))
                #else:
                    #args_in.append(x)
                #if y is not None:
                    #tmp.append(y)
                
        #return args_in + tmp
        
    #def ctype_def(self):
        #"""
        #The ctype type of this object
        #"""
        #if '_cached_ctype' not in self.__dict__:
            #self._cached_ctype = getattr(ctypes, self.proc['ret']['ctype'])
        
        #return self._cached_ctype
    
    #def _ctypes_to_return(self,args_out):
    
        #r = {}
        #self.args_out = {}
        
        #if self.save_args:
            ## Save arguments inside this object
            #for i,j in zip(self._arg_vars,args_out):
                #if 'out' in i.var['intent'] or i.var['intent']=='na': 
                    #r[i.name]=''
                    #if hasattr(j,'contents'):
                        #self.args_out[i.name]=j.contents
                    #else:
                        #self.args_out[i.name]=j
        #else:
            ## Copy arguments into a dict for returning
            #for i,j in zip(self._arg_vars,args_out):
                #if 'out' in i.var['intent'] or i.var['intent']=='na':
                    #r[i.name]=i.ctype_to_py_f(j)

        #return r
    
    #def __call__(self, *args):
        #args_in = self._args_to_ctypes(args)
        
        ## Capture stdout messages
        #with captureStdOut() as cs:        
            #if len(args_in) > 0:
                #res = self._call(*args_in)
            #else:
                #res = self._call()

        #if self._sub:
            #return self._ctypes_to_return(args_in)
        #else:
            #return self.returnPytype()(res)
            
    #def saveArgs(self,v=False):
        #""" Instead of copying arguments back we save them
        #inside the func object so we dont need to copy them
        #"""
        #if v:
            #self.save_args=True
        #else:
            #self.save_args=False
            
    #def returnPytype(self):
        #if '_cached_pytype' not in self.__dict__:
            #self._cached_pytype = getattr(__builtin__, self.proc['ret']['pytype'])
        
        #return self._cached_pytype
            
    #def __str__(self):
        #return str("Function: " + self.name)

    #def __repr__(self):
        #return self.__str__()

    #@property
    #def __doc__(self):
        #s = "Function: " + self.name + "("
        #if len(self._arg_vars) > 0:
            #s = s + ",".join([i._pname() for i in self._arg_vars])
        #else:
            #s = s + "None"
        #s = s + ")" + os.linesep+' '
        #s = s + "Args In: " + \
            #", ".join([i._pname()
                      #for i in self._arg_vars if 'in' in i.var['intent']]) + ' '+os.linesep+' '
        #s = s + "Args Out: " + \
            #", ".join([i._pname()
                      #for i in self._arg_vars if 'out' in i.var['intent']]) + ' '+os.linesep+' '
        #s = s + "Returns: "
        #if self.sub:
            #s = s + "None"
        #else:
            #s = s + str(self.pytype)
        #s = s + os.linesep+' '
        #return s

    #def __bool__(self):
        #return True
        
    #def __len__(self):
        #return 1
