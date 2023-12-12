#ifndef _Data
#define _Data

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdbool.h>
#include <cblas.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

/* Local types */
#ifdef SIMPLE
   #define MyType       float
   #define MyFFTcompl   fftwf_complex
   #define MyFFTCPUType fftwf_plan
#else
   #define MyType       double
   #define MyFFTcompl   fftw_complex
   #define MyFFTCPUType fftw_plan
#endif

/* General defines */
#define PIx2  6.283185307179586
#define dEPS  2.220446049250313e-16 // Like matlab for double
#define sEPS  2.2204460e-08         // Like matlab for float

/* Problem defines */
#define soft_window 5

/* For check errors */
#define OK       0
#define ErrNULL -1

#define CHECKERR(x) do { if((x)<0) { \
   printf("Error %d calling %s line %d\n", x, __FILE__, __LINE__);\
   return x;}} while(0)

#define CHECKNULL(x) do { if((x)==NULL) { \
   printf("NULL (when open file or  memory allocation calling %s line %d\n", __FILE__, __LINE__);\
   return ErrNULL;}} while(0)

#ifndef min
   #define min(x,y) ((x < y) ? x : y)
#endif

#ifndef max
   #define max(x,y) ((x > y) ? x : y)
#endif


void cBandPass(const int, const int, const int, const int, const int, const int, MyType*);
int       cDFT(const int, const int, const int, const int, const int, const MyType*, MyType*);
void    cGamma(const int, const MyType, MyType*);
void  cHamming(const int, MyType*);
int   cHammFFT(const int, const int, const int, const int, const int, const int, const int, const MyType*, const MyType*, MyType*);
void     cNorm(const int, MyType*);
int      cONMF(const int, const int, const int, const int, const MyType, const MyType, const int, MyType*, MyType*, MyType*);
void cSetZeros(const int, const int, const int, const int, MyType*);
void   cUpdate(const int, const MyType*, const MyType*, MyType*);
void  cUpdateL(const int, const int, const MyType, const MyType*, const MyType*, const MyType*, MyType*);
void   ctransp(const int, const int, const MyType*, MyType*);
void   ctransp2(const int, const int, const int, const MyType*, MyType*);
double  Ctimer(void);
int    cSmooth(const int, const int, const int, const int, const int, MyType*, const MyType*, MyType*, const MyType*, MyType*);
int   cFFTrows(const int, const int, const int, const int, const MyType*, MyType*);
int  cShiftFFT(const int, const int, MyType*); 

#endif
