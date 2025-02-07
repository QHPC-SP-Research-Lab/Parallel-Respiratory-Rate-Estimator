import os
import time
import numpy as np

from scipy.io     import wavfile   as wav
from numpy        import linalg    as LA
from numpy        import transpose as tp
from scipy        import linalg    as sLA
from random       import random
from time         import time
from RRprototypes import *

eps = np.finfo(np.double).eps

def NombresFicheros(ruta=os.getcwd()):
  contenido = os.listdir(ruta)
  ficheros  = []
  for fichero in contenido:
    if os.path.isfile(os.path.join(ruta, fichero)) and fichero.endswith('.wav'):
        ficheros.append(fichero)
  return ficheros

def NormalizaAudio(data):
    """
    Normaliza el audio según su formato de datos. Devuelve el audio normalizado en float64 (-1 a 1).
    """
    dtype = data.dtype
    if dtype == np.int16:
        normalized_data = data.astype(np.float64) / 32768.0
    elif dtype == np.int32:
        normalized_data = data.astype(np.float64) / 2147483648.0
    elif dtype == np.uint8:
        normalized_data = (data.astype(np.float64) - 128) / 128.0
    elif dtype == np.float32: 
        normalized_data = data.astype(np.float64)
    elif dtype == np.float64:
        normalized_data = data
    else:
        raise ValueError(f"Formato de audio no soportado: {dtype}")

    #print(f"Tipo de dato original: {data.dtype}")
    #print(f"Numero de muestras {data.shape[0]}")
    #print(f"Rango de valores normalizados: [{normalized_data.min()}, {normalized_data.max()}]")
    return normalized_data

def RR_estimator(dft_arr, K, fsh):
    step = fsh/(len(dft_arr[1,:])-1)
    f    = np.arange(-1*fsh/2,fsh/2+step, step,dtype=np.double)

    lim_in    = np.argmin(np.abs(f - 0.13)); #==>RR=6
    lim_sup   = np.argmin(np.abs(f - 0.38)); #==>RR=23
    lim_sup2  = np.argmin(np.abs(f - 1));    #==>RR=60
    f_window  = f[lim_in:lim_sup+1]
    f_window2 = f[lim_in:lim_sup2+1]
    
    max_val=np.amax(dft_arr[:,lim_in:lim_sup+1],axis=1)
    max_pos=np.argmax(dft_arr[:,lim_in:lim_sup+1],axis=1)
   
    hRR = np.argmax(max_val)
    f_hRR = f_window[max_pos[hRR]]
    f_parm = f_hRR/2
    
    detect = np.zeros((1,len(f)))
    dif_f2 = np.abs(f_window2-f_hRR);
    pos_parm2 = np.argmin(dif_f2)
    margen_vent2=int(np.round(0.02/(fsh/(len(dft_arr[1,:])-1))))
    
    detect[0,((lim_in)+pos_parm2)-margen_vent2:((lim_in)+pos_parm2)+margen_vent2+1]=np.sum(dft_arr[:,((lim_in)+pos_parm2)-margen_vent2:((lim_in)+pos_parm2)+margen_vent2+1],axis=0)
    
    f_parm1 = f_hRR/2
    dif_f1 = np.abs(f_window2-f_parm1);
    pos_parm1 = np.argmin(dif_f1)
    margen_vent1=int(np.round(0.02/(fsh/(len(dft_arr[1,:])-1))))
    
    detect[0,((lim_in)+pos_parm1)-margen_vent1:((lim_in)+pos_parm1)+margen_vent1+1]=np.sum(dft_arr[:,((lim_in)+pos_parm1)-margen_vent1:((lim_in)+pos_parm1)+margen_vent1+1],axis=0)
    
    f_parm3 = f_hRR*2
    dif_f3 = np.abs(f_window2-f_parm3);
    pos_parm3 = np.argmin(dif_f3)
    margen_vent3=int(np.round(0.02/(fsh/(len(dft_arr[1,:])-1))))
    
    detect[0,((lim_in)+pos_parm3)-margen_vent3:((lim_in)+pos_parm3)+margen_vent3+1]=np.sum(dft_arr[:,((lim_in)+pos_parm3)-margen_vent3:((lim_in)+pos_parm3)+margen_vent3+1],axis=0)
        
    Ep = np.sum(dft_arr[:,lim_in+pos_parm1-margen_vent1:lim_in+pos_parm1+margen_vent1+1])
    Et = np.sum(dft_arr[:,lim_in+pos_parm3-margen_vent3:lim_in+pos_parm3+margen_vent3+1])

    if f_parm1<0.13:
        RR=60/(1/f_hRR)
    elif Et>Ep:
        RR=60/(1/f_hRR)
    else:
        RR=60/(1/f_parm1)    
    return RR


def process(file_name, arg2):
   # Read signal (monophonic channel at 16 bits per sample)
   #iDfile  = wave.open(file_name, mode='rb')
   #buffer  = np.frombuffer(iDfile.readframes(iDfile.getnframes()), dtype=np.int16)
   #signalX = buffer.astype(np.double) / 32768.0
   #sr      = iDfile.getframerate()
   #iDfile.close()

   sr, buffer = wav.read(file_name)
   signalX    = NormalizaAudio(buffer)
   seconds    = buffer.shape[0] / sr

   L           = signalX.size
   B           = 1     # Normalization factor
   N           = 512   # Window size
   S           = 0.5   # Overlapping
   bases       = 40    # Number of bases
   K1          = 25    # Pre-learned spectral patterns: bases - 15 
   rowsNMF     = N + 1
   Hop_samples = int(np.round(S*N))
   noverlap_   = N-Hop_samples
   NFrames     = int(np.floor((L-N+(N*S))/Hop_samples))
   nfft_       = 2*N
   nIter       = int(arg2)
   GAMMA       = 1.0
   Fmin        = 300
   Fmax        = 2000
   lam         = 0.1
   np.random.seed(seed=(4095 % 2120))
   
   # You should initialize the first K1 rows with their pre-trained bases, and the rest of the rows randomly. 
   # This wrapper initializes the entire matrix randomly.
   # This is not the correct way to obtain an accurate estimation of the RR.
   W0 = np.random.uniform(low=eps, high=1.0-eps, size=(rowsNMF, bases)).astype(np.double)
   H0 = np.random.uniform(low=eps, high=1.0-eps, size=(bases, NFrames)).astype(np.double)

   W = np.copy(W0, order='F')
   H = np.copy(H0, order='F')

   iDFT = np.zeros((K1, NFrames), dtype=np.double, order='F')
   Time = np.zeros(6)

   error = ONMF(signalX, sr, S, N, noverlap_, NFrames, rowsNMF, nfft_, bases, sr, nIter, GAMMA, Fmin, Fmax, K1, W, H, iDFT, Time, 0)
   if error:
      RR = -100
   else:
      RR = RR_estimator(iDFT, K1, 1.0/(N*S/sr))

   return int(RR)
