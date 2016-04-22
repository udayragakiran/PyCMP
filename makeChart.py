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
# dataDir = 'D:\\Data\\Anping\\2016 chicken\\2016 4 6 chicken_Good'
dataDir = 'Z:\\Data\\OCT\\2016 Chicken\\2016 4 6 chicken_Good as representative'
# fileList = ['CM 1 20_44_16.pickle', 'CM 2 20_44_58.pickle']
  
from PyQt4 import QtCore, QtGui, uic


caption = "Choose filename"
# directory = os.getcwd()

app = QtGui.QApplication(sys.argv)

# dataDir = QtGui.QFileDialog.getExistingDirectory (None, caption, directory)
filterStr = ('*.pickle')
# fileList = QtGui.QFileDialog.getOpenFileNames (None, caption, directory, filterStr)
filepath = QtGui.QFileDialog.getOpenFileName(None, caption, dataDir, filterStr)

(dataDir, fileName) = os.path.split(filepath)

r = re.split(' ', fileName)
fileType = r[0]
number = r[1]
print('fileType=', fileType)

# filepath = os.path.join(dataDir, fileName)
print('loading ', filepath)

f = open(filepath, 'rb')
data = pickle.load(f)
f.close()

d = datetime.datetime.now()
timeStr = d.strftime('%Y-%m-%d %H_%M_%S')

if fileType == 'ABR':
    plotName = 'ABR %s %s' % (number, data.note)
    print('saving figure ', plotName)
    
    ABR.saveABRDataFig(data, data.params, dataDir, plotName, timeStr)
    

