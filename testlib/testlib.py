# from hopfield import Hopfield

import os
import matplotlib.pyplot as plt
import numpy as np
import random
from pprint import pprint
from tqdm.notebook import tqdm
from typing import Callable
from typing import Iterable

import itertools as it
import pandas as pd
import copy


class Person:
    def __init__(self, id: int, label: bool, r: np.ndarray, g: np.ndarray, b: np.ndarray, dtype=np.float32, sort: bool = False) -> None:
        self._dtype = dtype
        self._id = int(id)
        self._label = bool(label)
        self._data_list = dict()
                
        self._data_list['r'] = np.array(r, dtype=dtype)
        self._data_list['g'] = np.array(g, dtype=dtype)
        self._data_list['b'] = np.array(b, dtype=dtype)
        
        if sort:
            self._data_list['r'].sort()
            self._data_list['g'].sort()
            self._data_list['b'].sort()
        
        self.colors = [] 
        if len(self._data_list['r']):
            self.colors.append('r')
        if len(self._data_list['g']):
            self.colors.append('g')
        if len(self._data_list['b']):
            self.colors.append('b')
        
        
        
     
    def generate_bin_data(self, params: dict):
        """Generates self.bin_data and updates it

        Args:
            params (dict): 
                keys: 'r', 'g', 'b';
                values: dict of settings like:
                    'radius': uint
                    'bounds': tuple (min, max)
                    'precision': float
        """
        
        if len(params.keys())>3:
            raise ValueError("Only rgb filters allowed")
        
        self._data_bin = dict()
        
        for color, param in params.items():
            
            a, b = param['bounds']
            r = param['radius']
            precision = param['precision']
            
            length = int((b - a) // precision) + 1
            
            if color in self.colors:
                self._data_bin[color] = np.zeros(length)
                
                self._update_bin_filter(color, param)
            else:
                raise ValueError(f"This filter: {color} isn't in dataset")
            
            
            
            
    def _update_bin_filter(self, filter: str, params: dict):
        a, b = params['bounds']
        radius = params['radius']
        precision = params['precision']
        
        for dot in self._data_list[filter]:
            if dot > b or dot < a:
                continue
            
            dot = int((dot-a)// precision)        
            self._data_bin[filter][max(0, dot-radius) : min(dot+radius+1, len(self._data_bin[filter]))].fill(1)
            
    
    def _update_bin_filter_lazy(self, filter: str, params: dict):
        a, b = params['bounds']
        radius = params['radius']
        precision = params['precision']
        
        for dot in self._data_list[filter]:
            if dot > b or dot < a:
                continue
            
            dot = int((dot-a) // precision)        
            self._data_bin[filter][dot-radius] = 1
            self._data_bin[filter][dot+radius] = 1

                
    
            
    
    
            
            
            
          
        
        
        
        
        
        
        
    def get_bin_image(self, radius: int = 0, precision: float = 1e-4, up=1.1, down=0.8, filters: tuple = ("r", "g", "b")):
        pass
            
            
        
        

        

    
        
        





class Dataset:
    
    class Bin:
        """Binary 
        """
    
    def _max_id(self) -> int:
        max_id = 0
        if len(self._data)==0:
            return -1
        
        for i in self._data:
            max_id = max(max_id, i._id)
            
        return max_id
    

    def _load_dir(self, path: str, label: bool, sort: bool = True) -> None:
        start_id = self._max_id() + 1
        
        fnames = [(path+"/Red/"+i[:2]+"_Red.txt",
                   path+"/Green/"+i[:2]+"_Green.txt",
                   path+"/Blue/"+i[:2]+"_Blue.txt")
                  for i in [i for i in os.walk(path)][1][2]]
        
        update = np.empty(len(fnames), dtype=Person)
        
        for i in range(len(update)):
            with open(fnames[i][0],"r") as r_file, \
                 open(fnames[i][1],"r") as g_file, \
                 open(fnames[i][2],"r") as b_file: 
                      
                r = np.array([float(i) for i in r_file.readlines()[1:]])
                g = np.array([float(i) for i in g_file.readlines()[1:]])
                b = np.array([float(i) for i in b_file.readlines()[1:]])  

            update[i] = Person(start_id + i, label, r, g, b, sort=sort)
            
        self._data = np.append(self._data, update)    
        
    
    def __init__(self, path: str = None, posDir="BC", negDir="Control", dtype=np.float32, sort: bool = True, object=None) -> None:
        self._data = np.array([], dtype=Person)
        self._images = None
        if object is None:
            self._load_dir(path + '/' + posDir, True)
            self._load_dir(path + '/' + negDir, False)
        else:
            if type(object) is Dataset:
                # self._dtype = object._dtype
                # self._data = object._data
                raise NotImplementedError

            elif type(object) is np.ndarray:
                self._dtype = object.dtype
                self._data = object.copy() # not deep copy!!!
            else:
                raise TypeError
        
        # self._images
        
     
    def get_bin_images(self, radius: int = 0, precision: float = 1e-4, up=1.1, down=0.8, filters: tuple = ("r", "g", "b"), dtype=np.float32):           
            pass
        
    def __getitem__(self, key):
        if isinstance(key, slice):
            pass
        else:
            pass
            
            
        

            
            
            
        
        
        
