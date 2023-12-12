# Parallel-Respiratory-Rate-Estimator
## Overview
This project involves audio processing for Parallel-Respiratory-Rate-Estimator using a combination of C and Python.

## Prerequisites
Ensure that you have the following:
- **C Compiler:** Install a C compiler on your system. If you don't have one installed, you can use GCC.
- **BLAS/LAPACK Library:** Obtain and install a distribution of BLAS/LAPACK. This library is used for numerical linear algebra operations.
- **FFTW Library:** Download and install the FFTW library, which is used for fast Fourier transform computations.
- **Python 3:** Make sure you have Python 3 installed on your system.

## Compilation
Use the following steps to compile the C code:
bash
make
Make sure to link against the BLAS, LAPACK, and FFTW libraries, and the math library.

## Execution
To run the project, execute the compiled binary along with the Python script:
bash
python RRmain.py input_file.wav 100
Replace 'input_file.wav' with the path to your input audio file and adjust '100' to the desired number of iterations for the ONMF factorization.

## Note
Ensure that the input audio files are in WAV format, have a monochannel configuration, and use 16 bits per sample for accurate processing.

Feel free to reach out for any issues or questions!

Feel free to modify this template according to the specifics of your project.

