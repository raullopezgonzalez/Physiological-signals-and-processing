import numpy as np
from scipy.signal import gaussian


# taking from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

 #   if x.ndim != 1:
 #       raise ValueError, "smooth only accepts 1 dimension arrays."

 #   if x.size < window_len:
 #       raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


#    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y[int((window_len/2-1)):-int((window_len/2))]


def gaussian_smoothing(x,w_len = 11,sigma=3):
    """
    Smoothing using a gaussian window with full-width half-maximum sigma in samples
    """
    
    gauss_window = gaussian(w_len,sigma)
    
    s=numpy.r_[x[w_len-1:0:-1],x,x[-2:-w_len-1:-1]]#get signal
    y=numpy.convolve(gauss_window/gauss_window.sum(),s,mode='valid')
    return y[int((w_len/2-1)):-int((w_len/2))]

def maxima_minima(x):
    """
    Get local maxima and minima from a not too much noisy signal
    """
    
    #1. first derivative

    y_prime = np.diff(y_d)
    #2. sign of the first derivative

    sig_y_prime = np.sign(y_prime)
    #3. derivative of the sign of the first derivative

    y_two_prime = np.diff(sig_y_prime)
    
    #4. find positive and negative point

    maxima = np.squeeze(np.where(y_two_prime<0))
    minima = np.squeeze(np.where(y_two_prime>0))
    
    return maxima, minima


def exp_beat_detection(ecg,fs,Tr = .180,a = .7,b = 0.999):
    """ input ecg: ecg to be detected
              Tr: refractory period in seconds. Default 180 ms
              a: correction fraction of the threshold with respecto to the
              maximun of the R peak. Default 0.7
              b: exponential decay. Default 0.999
              
        output beat: beat detected
               th: threshold function
               qrs_index: qrs index  
    """
    Tr = np.floor(Tr*fs)
    end = len(ecg) #end point
    maximum_value = np.max(ecg/3.)
    
    beat = np.zeros((end,1))
    qrs_index = []
    th = []
    th.append(a*max(ecg)) # threshold initialization 
    
    detect = Tr*(-1) + 1 #variable that controls the refractory period
    
    for k in range(1,end):
        #for over all the points in the ECG
        if (ecg[k] > th[k-1]) & (k > (detect + Tr)):
            detect = k #new refractory period
            qrs_index.append(k)
            beat[k] = maximum_value
            
        #if no detection, update the threshold
        update_th = np.max((a*ecg[k],b*th[k-1]))
        th.append(update_th)
        
    return beat, th, qrs_index

def r_peak_detection(ecg_filtered,ecg_original,fs,beat,th,qrs_index,Tr = .180):
    """ r-peak detection in the original ecg. Once, the QRS is detected in the
    filtered ecg, this functions allows to estimate de position of the r-peak
    
    inputs ecg_filtered: ecg filtered on which the qrs detection is performed.
           ecg_original: original ecg
           fs: sampling frequency.
           beat: qrs detections
           th: threshold function that allowed the detection of qrs complexes
           qrs_index: indices of the qrs in the filtered ecg.
           
    outputs
            r_peaks: position of the r_peaks on the original ecg
            rr: RR-interval time series
    """    
    Tr_s = np.round(Tr * fs)
    r_peaks = np.zeros(len(ecg_filtered))

    for idx in qrs_index:
        
        #build an r-peak search window around the qrs_complex detected
        window = range(idx-3,idx + int(Tr_s/2))
        
        if (window[0] > 0) & (window[-1] < len(ecg_filtered)):
            beat_original = ecg_original[window]
            idx_rpeak = np.argmax(beat_original)
            
            r_peaks[idx - 3 + idx_rpeak] = 1
    #end of for        
    r_peak = np.nonzero(r_peaks)[0]
    rr = (np.diff(r_peak)/fs) * 1000
    return r_peak, rr