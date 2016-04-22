# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:31:38 2015

@author: OHNS
"""
import numpy as np
from enum import Enum
import re
import os
import datetime
import struct
import pickle
import traceback
import sys
import xlsxwriter

        
def initExcelWorkbook(filepath):
    
    workbook = xlsxwriter.Workbook(filepath)
    
    return workbook
    
def initExcelSpreadsheet(workbook, name, number, timeStr, note):
    wsname =  name + ' ' + str(number) 
    ws = workbook.add_worksheet(wsname)
    
    ws.write(0, 0, 'Note')
    ws.write(0, 1, note)
    ws.write(1, 0, 'Timestamp')
    ws.write(1, 1,  timeStr)
    
    return ws
    
def initSaveDir(saveOpts, protocolName):
    baseDir = saveOpts.saveBaseDir
        
    d = datetime.datetime.now()
    timeStr = d.strftime('%Y-%m-%d %H_%M_%S')
    saveDir = timeStr + ' ' + protocolName
        
    if not os.path.exists(saveDir):   # create directory if it does not exist
        os.makedirs(saveDir)

    
def writeExcelFreqAmpHeader(ws, freq, amp, row=0, col=1):
    # write amplitude header in first row
    r = row
    c = col
    for a in amp:
        c += 1
        ws.write(r, c, a)

    #write frequency header down col
    c = col
    for f in freq:
        r += 1
        ws.write(r, c, f)
        
def writeExcel2DData(ws, data, row=1, col=1):
    for r in range(0, data.shape[0]):
        for c in range(0, data.shape[1]):
            d = data[r,c]
            
            if not (np.isinf(d) or np.isnan(d)):
                ws.write(r+row, c+col, d)

class blankRecord:
    pass

class SaveOpts:
    def __init__(self):
        self.note = ''
        self.saveBaseDir = ''
        
        self.saveRaw = False
        
    def __repr__(self):
        s = 'saveBaseDir= %s note= %s saveRaw = %s' % (self.saveBaseDir, self.note,  self.saveRaw)
        
        return s

def readMicResponseFile(filePath):
    f = open(filePath)
    lines = f.readlines()    
    f.close()
    
    nameArray = []
    respArray = []
    idx = 0
    while idx < len(lines):
        if lines[idx] != '':
            try:
                name = lines[idx]
                name = name[:-1]  # re3move newline
                print(name)
                idx += 1
                freq_str = lines[idx]
                print(freq_str)
                idx += 1
                dB_str = lines[idx]
                print(dB_str)
                # this executes the frequency string which should be of the form F = [1, 20, 500], so that F should now be defined
                F = eval(freq_str)   
                # execute the db string so that dB is now defined
                dB = eval(dB_str)
                resp = np.vstack((F, dB))
                nameArray.append(name)
                respArray.append(resp)
            except Exception:
                traceback.print_exc(file=sys.stdout)
                DebugLog.log('CMPCommon.readMicResponseFile: error parsing file at line %d' % idx)
                
        idx += 1 # increment index
                
    return nameArray, respArray
        
    

# tests    
if __name__ == "__main__":   
    pass
    