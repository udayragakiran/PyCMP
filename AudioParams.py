# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:32:28 2015

@author: OHNS
"""

import numpy as np

class Speaker(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3
    
class FreqSpacing(Enum):
    LINEAR = 1
    LOGARITHMIC = 2
    CUSTOM = 3
    
class AudioOutputParams:
    def __init__(self):
        self.freq = np.array([ [1e3], [1e3] ])     # array frequencies in Hz, first index is speaker num, second  is frequency index,
        self.amp = [ 80 ]           # array amplitudes in decibels
        self.stimDuration = 50      # duration of stim in milliseconds
        self.trialDuration = 50     # duration of entire trialin milliseconds 
        self.speakerSel = Speaker.LEFT    # which speaker or speakers to use
        self.stimType = AudioStimType.PURE_TONE
        self.stimOffset = 0         # offset ofthe stim in milliseconds
        self.stimEnvelope = 1       # envelope of the stim stim milliseconds
        self.useAmpDepTrialDuration = False
        self.ampDepTrialDurationArray = []
        self.numTrials = 1
        self.downsample = 0         # used for FPGA downsampling  
        self.freqSpacing = FreqSpacing.LINEAR
        self.customSoundir = ''
        
    def getTrialDuration(self, amp):
        if(self.useAmpDepTrialDuration):
            aIdx = self.amp.index(amp)
            self.ampDepTrialDurationArray[aIdx]    
        else:
            return self.trialDuration
    
    def getNumSpeakers(self):
        numSpk = 1
        if self.speakerSel == Speaker.BOTH:
            numSpk = 2
       
        return numSpk
       
    def getNumFrequencies(self):
        return self.freq.shape[1]
        
    def __repr__(self):
        freqStr = ''
        for n in range(0, len(self.freq[0, :])):
            freqStr = freqStr + ' %0.3f' % self.freq[0, n]
        s = 'freq= %s' % freqStr 
        
        ampStr = ''
        for n in range(0, len(self.amp)):
            ampStr = ampStr + ' %0.3f' % self.amp[n]
        s = s + '\namp= %s' % ampStr 
        s = s + '\nstimDuration= %0.3f trialDuration= %0.3f ' % (self.stimDuration, self.trialDuration)
        s = s + '\nspeakerSel= %s stimType= %s ' % (repr(self.speakerSel), repr(self.stimType))
        s = s + '\nstimOffset= %0.3f stimEnvelope = %0.3f' % (self.stimOffset, self.stimEnvelope)
        s = s + '\nuseAmpDepTrialDuration= %s ' % (repr(self.useAmpDepTrialDuration))
        s = s + '\nampDepTrialDurationArray= %s ' % (repr(self.ampDepTrialDurationArray))
        s = s + '\nnumTrials= %d downsample= %d' % (self.numTrials, self.downsample)
        
        return s
        
if __name__ == "__main__":
    audioParams = AudioOutputParams()
    freqStart = 4
    freqEnd = 10
    freqSteps = 7
    freq = np.linspace(freqStart, freqEnd, freqSteps)
    audioParams.freq = np.tile(freq, (2, 1))
    audioParams.amp = np.linspace(10, 80, 9)
    print(repr(audioParams))