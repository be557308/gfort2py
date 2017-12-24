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
            # Cat a objects ctype+pytype for the key (logical are not C_bool but c_int so must distingush from normal int's)
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
		self._init()
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
		
	
	def _init(self):
		pass
			

	
	def _getType(self,obj):
		
		if 'dt' in obj:
			pass
		elif 'array' in obj:
			pass
		elif 'param' in obj:
			pass
		elif 'var' in obj:
			pass
				
				
	def __dir__(self):
		return list(self._mod_vars.keys())+list(self._param.keys())+list(self._funcs.keys())
           
           
    # TODO: getattr and setattr need to be refactored into small function
           
	def __getattr__(self,key):
		if key in self.__dict__:
			return self.__dict__[key]
		
		key = key.lower()
		x = None
		
		if self._initilised:
			if key in self._mod_vars :
				y = self._mod_vars[key]
				try:
					#Allready loaded variable
					x = y['gf']
				except KeyError:
					# Variable not loaded yet
					v = y['var']
					y['gf'] = _map2gf(y)(name=y['name'],mangled_name=y['mangled_name'],**v)
					x = y['gf']
			elif key in self._param:
				y = self._param[key]
				try:
					#Allready loaded variable
					x = y['gf']
				except KeyError:
					# Variable not loaded yet
					v = y['param']
					y['gf'] = _map2gf(y)(name=y['name'],param=True,mangled_name=y['mangled_name'],**v)
					x = y['gf']
			elif key in self._funcs:
				y = self._funcs[key]
				try:
					#Allready loaded variable
					x = y['gf']
				except KeyError:
					# function not loaded yet
					v = y['arg']
					fv = [_map2gf(i['var'])(**i['var']) for i in v]
					name = [i['name'] for i in v]
					opt = [i['var']['optional'] for i in v]
					intent = [i['var']['intent'] for i in v]
					y['gf'] = func.fFunc(name=y['name'],mangled_name=y['mangled_name'],
										arg_return=y['proc']['ctype'],
										arg_names=name,arg_fvar=fv,
										arg_opts=opt,arg_intents=intent)
					x = y['gf']
			else:
				raise KeyError("Key "+str(key)+" does not exist")
				
		return x
		
	def __setattr__(self,key,value):

		key = key.lower()
		x = None
		if self._initilised:
			if key in self._mod_vars :
				y = self._mod_vars[key]
				try:
					#Allready loaded variable
					x = y['gf']
				except KeyError:
					# Variable not loaded yet
					v = y['var']
					y['gf'] = _map2gf(y)(name=y['name'],mangled_name=y['mangled_name'],**v)
					
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
	
