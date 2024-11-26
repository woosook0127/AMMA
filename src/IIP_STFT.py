import numpy as np
from scipy.signal import get_window

class IIPSTFT:
    
    # 생성자
    def __init__(self, winL,nfft,nShift,fs):
        self.winL = winL
        self.nfft = nfft
        self.nShift = nShift
        self.fs = fs
        
        # By-product
        shiftdiv = winL // nShift # 2 or 4
        self.nhfft = nfft // 2 + 1 # 257
        
        if shiftdiv == 2:
            self.win = np.sin(np.pi * (np.arange(winL) + 0.5) / winL)
        elif shiftdiv == 4:
            self.win = np.sqrt(2/3) * get_window('hann', winL)


        

    # 메서드
    def STFT(self,x_wav):
        # x_wav: input signal nsample x nch
        nSample, nch = x_wav.shape
        nFrame = (nSample + self.winL)//self.nShift - 2
        
        x_padd = np.zeros((nSample + self.winL + self.nfft, nch))
        x_padd[self.winL:self.winL + nSample, :] = x_wav
        
        X = np.zeros((nch, self.nhfft, nFrame), dtype=complex) 
        for frame in range(nFrame):
            x_frame = x_padd[(frame+1) * self.nShift : (frame+1) * self.nShift + self.winL, :]
            for ch in range(nch):
                X[ch, :, frame] = np.fft.rfft(x_frame[:, ch] * self.win, self.nfft)

            
        return X
    
    def iSTFT(self, X, nSample):
        nch, _, nFrame = X.shape
        y_padd = np.zeros((nSample + self.winL + self.nfft, nch))
        for frame in range(nFrame):
            for ch in range(nch):
                y_tmp1 = np.fft.irfft(X[ch,:, frame], self.nfft)
                y_padd[(frame+1) * self.nShift : (frame+1) * self.nShift + self.winL,ch] += y_tmp1[:self.winL] * self.win
        y = y_padd[self.winL:self.winL + nSample, :]
        return y
