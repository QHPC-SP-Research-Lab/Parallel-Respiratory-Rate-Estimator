import ctypes

from numpy.ctypeslib import ndpointer as ND

Lib = ctypes.cdll.LoadLibrary('./RRmain.so')

ONMF          = Lib.ONMF
ONMF.restype  = ctypes.c_int
ONMF.argtypes = [ND(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int,
                 ctypes.c_int, ND(ctypes.c_double, flags='F'), ND(ctypes.c_double, flags='F'), ND(ctypes.c_double, flags='F'),
                 ND(ctypes.c_double), ctypes.c_bool]
