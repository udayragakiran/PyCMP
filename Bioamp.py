# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:45:03 2015

@author: OHNS
"""
import numpy as np
import re   # regular expressions

class Bioamp:
    def __init__(self):
        self.gain = 10e3
        self.daqChan = "Dev1/ai1"
        
        self.DAQdevice = 'Dev1'                    
    
    def decodeFromString(self, audio_str):
        lines = re.split('\n', audio_str)  # break up lines into array
        for s in lines:
            x = re.split('=', s)
            if(len(x) < 2):
                continue
            fld = x[0]
            val = x[1]
            if(fld == 'DAQ Channel'):
                self.daqChan = val
            elif(fld == 'Gain'):
                self.gain = float(val)
        
def readBioampConfig(filepath):    
    bioamp = Bioamp()
    
    f = open(filepath, "r")
    txt = f.read()
    f.close()
    bioamp.decodeFromString(txt)
    
    return bioamp