# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:27:04 2016

@author: OHNS
"""

import CM
import ABR
import DPOAE
import pickle
import os
import CMPCommon
import re
import datetime
import traceback
import sys

#dataDir = 'D:\\Data\\Alix\\20160329 CBAF1 P'
dataDir = 'D:\\Data\\Anping\\2016 chicken\\2016 4 6 chicken_Good'
fileList = ['CM 1 20_44_16.pickle', 'CM 2 20_44_58.pickle']
  
    
from PyQt4 import QtCore, QtGui, uic


caption = "Choose directory"
directory = os.getcwd()

app = QtGui.QApplication(sys.argv)

# dataDir = QtGui.QFileDialog.getExistingDirectory (None, caption, directory)
filterStr = ('*.pickle')
fileList = QtGui.QFileDialog.getOpenFileNames (None, caption, dataDir, filterStr)

all_data = []
all_types = []
all_numbers = []
for filepath in fileList:
    path, fName = os.path.split(filepath)
    print("loading ", fName)
    r = re.split(' ', fName)
    all_types.append(r[0])
    all_numbers.append(int(r[1]))
    filepath = os.path.join(dataDir, fName)
    f = open(filepath, 'rb')
    data = pickle.load(f)
    f.close()
    all_data.append(data)

d = datetime.datetime.now()
timeStr = d.strftime('%Y-%m-%d %H_%M_%S')
filename = 'PyCMP ' + timeStr + '.xlsx'
filepath = os.path.join(dataDir, filename)
wb = CMPCommon.initExcelWorkbook(filepath)

for data, name, number in zip(all_data, all_types, all_numbers):
    try:
        saveOpts = CMPCommon.SaveOpts()
        print('writing ', name, ' ', number, ' to Excel spreadsheet')
        
        if name == 'CM':
            saveOpts.saveTracings = data.tracings is not None
                
            timeStr = data.timeStamp
            ws = CMPCommon.initExcelSpreadsheet(wb, name, number, timeStr, data.note)
            CM.saveCMDataXLS(data, data.trialDuration, data.trialReps, ws, saveOpts)
        elif name == 'DPOAE':
            saveOpts.saveMicData = data.mic_data is not None
            saveOpts.saveMicFFT = data.mic_fft_mag is not None
            
            timeStr = data.timeStamp
            ws = CMPCommon.initExcelSpreadsheet(wb, name, number, timeStr, data.note)
            DPOAE.saveDPOAEDataXLS(data, data.trialDuration, ws, saveOpts)
        elif name == 'ABR':
            saveOpts.saveTracings = data.tracings is not None
            
            ABRparams = data.params
            
            timeStr = data.timeStamp
            ws = CMPCommon.initExcelSpreadsheet(wb, name, number, timeStr, data.note)
            ABR.saveABRDataXLS(data, ABRparams, ws, saveOpts)
        #print(data.__dict__)
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        
print('closing workbook')
wb.close()
print('done')
    
    

