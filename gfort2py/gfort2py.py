# SPDX-License-Identifier: GPL-2.0+

from __future__ import print_function
try:
	import __builtin__
except ImportError:
	import builtins as __builtin__

import ctypes
import pickle
import numpy as np
import errno

from . import var
from . import utils
from . import arrays
from . import functions as func
from . import dt
from . import parseMod as pm

WARN_ON_SKIP=False

#https://gcc.gnu.org/onlinedocs/gcc-6.1.0/gfortran/Argument-passing-conventions.html

_map2ctype = {
            # Cat a objects ctype+pytype for the key (logical are not C_bool but c_int so must distinguish from normal int's)
            'c_intint':var.fInt,
            'c_int32int':var.fInt,
            'c_int64int':var.fLongInt,
            'c_floatfloat':var.fSingle,
            'c_doublefloat':var.fDouble,
            'c_longdoublequad':var.fQuad,
            'c_char_pstr':var.fChar,
            'c_intbool':var.fLogical,
            'c_floatcomplex':var.fSingleCmplx,
            'c_doublecomplex':var.fDoubleCmplx,
            'c_longdoublecomplex':var.fQuadCmplx
}

def _map2gf(x):
	r = None
	if 'param' in x:
		v = x['param']
		r = _map2ctype[v['ctype']+v['pytype']]
	elif 'var' in x:
		if 'array' in x['var']:
			r = arrays.fArray
		elif 'dt' in x['var']:
			r = dt.fDT
		else: # single variables
			v = x['var']
			r = _map2ctype[v['ctype']+v['pytype']]
	elif 'sub' in x:
		if not x['sub']:
			r = _map2ctype[x['ret']['ctype']+x['ret']['pytype']]()
	return r




class fFort(object):
	_initilised = False
	def __init__(self, libname, ffile, rerun=False):
		self._lib = ctypes.CDLL(libname)
		self._all_names=[]
		self._libname = libname
		self._fpy = pm.fpyname(ffile)
		self._load_data(ffile, rerun)
		utils.loadLib(self._libname)
		
		self._init_dt_def()
		self._initilised = True
	
	def _load_data(self, ffile, rerun=False):
		try:
			f = open(self._fpy, 'rb')
		# FileNotFoundError does not exist on Python < 3.3
		except (OSError, IOError) as e: 
			if e.errno != errno.ENOENT:
				raise
			pm.run(ffile,save=True)
		else:
			f.close()
	
		with open(self._fpy, 'rb') as f:
			self.version = pickle.load(f)
			if self.version == 2:
				self._mod_data = pickle.load(f)
	
				if self._mod_data["checksum"] != pm.hashFile(ffile) or rerun:
					self._rerun(ffile)
				else:
					self._mod_vars = pickle.load(f)
					self._param = pickle.load(f)
					self._funcs = pickle.load(f)
					self._dt_defs = pickle.load(f)
			else:
				self._rerun(ffile)
				
				
	def _rerun(self,ffile):
		x = pm.run(ffile,save=True,unpack=True)
		self._mod_data = x[0]
		self._mod_vars = x[1]
		self._param = x[2]
		self._funcs = x[3]
		self._dt_defs = x[4]			
			
	def _init_param(self,y):
		v = y['param']
		gf = _map2gf(y)
		return gf(name=y['name'],mangled_name=y['mangled_name'],param=True,**v)
		
	def _init_var(self,y):
		v = y['var']
		gf = _map2gf(y)
		if 'array' in v:
			return gf(name=y['name'],mangled_name=y['mangled_name'],**v['array'])
		elif 'dt' in v:
			return gf(name=y['name'],mangled_name=y['mangled_name'],dt_name=v['dt']['name'])
		else:
			return gf(name=y['name'],mangled_name=y['mangled_name'],**v)
				
		
	def _init_func(self,y):
		v = y['arg']
		fv = [_map2gf(i)(**i['var']) for i in v]
		name = [i['name'] for i in v]
		opt = [i['var']['optional'] for i in v]
		intent = [i['var']['intent'] for i in v]
		ar = _map2gf(y['proc'])
		return func.fFunc(name=y['name'],mangled_name=y['mangled_name'],
							arg_return=ar,
							arg_names=name,arg_fvar=fv,
							arg_opts=opt,arg_intents=intent)
				
				
	def _init_dt_def(self):
		for key, value in self._dt_defs.items():
			dt._dictDTDefs[key] = {}
			args = value['dt_def']['arg']
			dt._dictDTDefs[key]['keys'] = [i['name'] for i in args]
			dt._dictDTDefs[key]['key_types'] = [getattr(ctypes,i['var']['ctype']) for i in args]
			dt._dictDTDefs[key]['key_sizes'] = [int(i['var']['bytes']) for i in args]
			dt._dictDTDefs[key]['key_fvars'] = [_map2gf(i) for i in args]
			dt._dictDTDefs[key]['key_extras'] = [i['var']['dt']['name'] if 'dt' in i['var'] else None for i in args]
			
				
	def __dir__(self):
		return list(self._mod_vars.keys())+list(self._param.keys())+list(self._funcs.keys())
           
	def __getattr__(self,key):
		if key in self.__dict__:
			return self.__dict__[key]
		
		key = key.lower()

		if self._initilised:
			if key in self._mod_vars :
				y = self._mod_vars[key]
				if 'gf' not in y:
					y['gf'] = self._init_var(y)
			elif key in self._param:
				y = self._param[key]
				if 'gf' not in y:
					y['gf'] = self._init_param(y)
			elif key in self._funcs:
				y = self._funcs[key]
				if 'gf' not in y:
					y['gf'] = self._init_func(y)
			else:
				raise KeyError("Key "+str(key)+" does not exist")
				
			return y['gf']
		
	def __setattr__(self,key,value):

		key = key.lower()
		if self._initilised:
			if key in self._mod_vars :
				y = self._mod_vars[key]
				if 'gf' not in y:
					y['gf'] = self._init_var(y)
				y['gf'].value = value
			elif key in self._param:
				raise AttributeError("Can't alter a parameter")
			elif key in self._funcs:
				raise AttributeError("Can't set a procedure")
			else:
				self.__dict__[key] = value
		else:
			self.__dict__[key] = value		
		 

		                   
	
	#def _init(self):
		#self._listVars = []
		#self._listParams = []
		#self._listFuncs = []
		
		#for i in self._mod_vars:
			#self._all_names.append(i['name'])
		
	
		#for i in self._param:
			#self._all_names.append(i['name'])
			
		#for i in self._funcs:
			#self._all_names.append(i['name'])
			
		#self._all_names = set(self._all_names)
			
		#for i in self._dt_defs:
			#i['name']=i['name'].lower().replace("'","")
			
		#self._init_dt_defs()
		
		#for i in self._mod_vars:
			#self._init_var(i)
	
		#for i in self._param:
			#self._init_param(i)
	
		## Must come last after the derived types are setup
		#for i in self._funcs:
			#self._init_func(i)
	
	#def _init_var(self, obj):
		#x=None
		#if obj['var']['pytype'] == 'str':
			#x = fStr(self._lib, obj)
		#elif obj['var']['pytype'] == 'complex':
			#x = fComplex(self._lib, obj)
		#elif 'dt' in obj['var'] and obj['var']['dt']:
			#x = fDerivedType(self._lib, obj)
		#elif 'array' in obj['var']:
			#array = obj['var']['array']['atype'] 
			#if array == 'explicit':
				#x = fExplicitArray(self._lib, obj)
			#elif array == 'alloc' or array == 'pointer':
				#x = fDummyArray(self._lib, obj)
			#elif array == 'assumed_shape':
				#x =  fAssumedShape(self._lib, obj)
			#elif array == 'assumed_size':
				#x = fAssumedSize(self._lib, obj)
		#else:
			#x = fVar(self._lib, obj)
	
		#if x is not None:
			#self.__dict__[x.name] = x
		#else:
			#print("Skipping init "+obj['name'])
	
	#def _init_param(self, obj):
		#if obj['param']['pytype']=='complex':
			#x = fParamComplex(self._lib, obj)
		#elif obj['param']['array']:
			#x = fParamArray(self._lib, obj)
		#else:
			#x = fParam(self._lib, obj)
	
		#self.__dict__[x.name] = x
	
	#def _init_func(self, obj):
		#x = fFunc(self._lib, obj)
		#self.__dict__[x.name] = x
		
	#def _init_dt_defs(self):
		#all_dt_defs=self._dt_defs
		
		#completed = [False]*len(all_dt_defs)
		## First pass, do the very simple stuff (things wih no dt's inside them)
		#for idx,i in enumerate(all_dt_defs):
			#flag=True
			#for j in i['dt_def']['arg']:
				#if 'dt' in j['var']:
					#flag=False
			#if flag:
				#_dictAllDtDescs[i['name']]=_DTDesc(i)
				#completed[idx]=True
				
		#progress = True
		#while True:     
			#if all(completed):
				#break
			#if not progress:
				#break
			#progress=False
			#for idx,i in enumerate(all_dt_defs):
				#if completed[idx]:
					#continue
				#flag=True
				#for j in i['dt_def']['arg']:
					#if 'dt' in j['var']:
						#if j['var']['dt']['name'] not in _dictAllDtDescs:
							#flag=False
							
				##All elements are either not dt's or allready in the alldict
				#if flag:
					#progress = True
					#_dictAllDtDescs[i['name']]=_DTDesc(i)
					#completed[idx]=True
		
				   
		## Anything left not completed is likely to be a recursive type
		#for i,status in zip(all_dt_defs,completed):
			#if not status:
				#_dictAllDtDescs[i['name']]=getEmptyDT(i['name'])
		
		
		## Re-do the recurivse ones now we can add the empty dt's to them
		#for i,status in zip(all_dt_defs,completed):
			#if not status:
				#_dictAllDtDescs[i['name']] = _DTDesc(i)
	
	#def __getattr__(self, name):
		#if name.lower() in self.__dict__:
			#return self.__dict__[name.lower()]
	
		#if '_all_names' in self.__dict__:
			#if name.lower() in self._all_names:
				#return self.__dict__[name.lower()].get()
	
		#raise AttributeError("No variable " + name)
	
	#def __setattr__(self, name, value):
		#if '_all_names' in self.__dict__:
			#if name in self._all_names:
				#self.__dict__[name].set_mod(value)
				#return
	   
		#self.__dict__[name] = value
		#return
	
