#include "RRdefines.h"


double Ctimer(void)
{
  struct timeval tm;
  gettimeofday(&tm, NULL);
  return tm.tv_sec + tm.tv_usec/1.0E6;
}


void cSetZeros(const int n, const int r, const int f0, const int f1, MyType *v)
{
  int i, j, tmp; 

  for(i=0; i<n; i++)
  {
    tmp = i*r;
    for(j=f0; j<f1; j++)
      #ifdef SIMPLE
        v[tmp+j] = sEPS;
      #else
        v[tmp+j] = dEPS;
      #endif
  }
}

// Slower than cSetZeros (suitable for long audio files) using Xavier.
void cSetZerosslow(const int n, const int r, const int f0, const int f1, MyType *v)
{
  int i;
  size_t itmp=(f1-f0)*sizeof(MyType);

  for(i=f0; i<f1; i++)
    #ifdef SIMPLE
      v[i] = sEPS;
    #else
      v[i] = dEPS;
    #endif
  for(i=1; i<n; i++)
    memcpy(&v[i*r+f0], &v[f0], itmp);
}


void cBandPass(const int n, const int r, const int N, const int sr, const int Fmin, const int Fmax, MyType* X)
{
  int fcut1 = (int)(ceil(Fmin / ((sr/2.0) / N)));
  int fcut2 = (int)(ceil(Fmax / ((sr/2.0) / N)));

  cSetZeros(n, r, 0,       fcut1, X);
  cSetZeros(n, r, fcut2-1, r,     X);
}


void cHamming(const int n, MyType *v)
{
  int    i, N;
  MyType tmp=PIx2/(MyType)(n-1);

  if ((n%2) == 0) { N=n;} else { N=n+1; }
  for(i=0; i<N/2; i++) 
  {
    #ifdef SIMPLE
      v[i]=v[n-i-1]= 0.54 - 0.46*cosf(i*tmp);
    #else
      v[i]=v[n-i-1]= 0.54 - 0.46* cos(i*tmp);
    #endif
  }
}


void cNorm(const int n, MyType *v)
{
  MyType sum;

  #ifdef SIMPLE
    sum=cblas_sasum(n, v, 1);
    cblas_sscal(n, (1.0/sum), v, 1);
  #else
    sum=cblas_dasum(n, v, 1);
    cblas_dscal(n, (1.0/sum), v, 1);
  #endif
}

// Slower than cNorm (suitable for long audio files) using Xavier.
void cNormslow(const int n, MyType *v)
{
  int i;
  MyType sum=.0;
  
  #pragma omp parallel for reduction(+:sum)
    for(i=0;i<n;i++) sum+=v[i];

  #pragma GCC ivdep
    for(i=0;i<n;i++) v[i] /= sum;
}

// Right now, it is not used.
void cGamma(const int n, const MyType gamma, MyType *v)
{
  int i;

  for(i=0; i<n; i++)
    #ifdef SIMPLE
      v[i] = powf(v[i], gamma);
    #else
      v[i] =  pow(v[i], gamma);
    #endif
}


void cUpdate(const int n, const MyType *Num, const MyType *Den, MyType *Res)
{
  int i;
  
  // Faster than "#pragma GCC ivdep" or "#pragma omp simd" for long audio files using Xavier.
  #pragma omp parallel for simd
  for(i=0; i<n; i++)
    Res[i] = Res[i] * Num[i] / Den[i];
}


void cUpdateL(const int F, const int K2, const MyType lamda, const MyType *Num, const MyType *Den, const MyType *Flam, MyType *W)
{
  int i, j, tmp;

  // Faster than "#pragma omp parallel for simd", K2 is too small  (K2=15 right now)
  for(i=0;i<K2;i++)
  {
    tmp = i*F;
    #pragma omp simd
    for(j=0;j<F;j++)
      W[tmp+j] = (W[tmp+j] * (Num[tmp+j] + lamda * W[tmp+j])) / (Den[tmp+j] + Flam[j]);
  }
}


int cHammFFT(const int winSize, const int Overlap,  const int nFrames,   const int rowsNMF, const int fftSize, 
             const int Threads, const int planType, const MyType *audio, const MyType *vHamming, MyType *rSNMF)
{
  int          k;
  MyFFTcompl   *xFFT=NULL;
  MyFFTCPUType *planFFT=NULL;

  /* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
  /* It will be done in-place, using the same vector for input and output. This save    */
  /* memory (512*8*maxThreads or 1024*8*maxThreads bytes) depending on when it's called */
  /*                                                                                    */
  /* We use this form of parallelism (several concurrent FFTs) because the compilation  */
  /* with the parallel version of FFTw throwing bad times (the size?)                   */
  /*                                                                                    */
  /* FFTw API functions are NOT thread-safe, except fft_execute(). Therefore, we should */
  /* declare/release everything outside the "#parallel omp parallel". It is what it is! */
  /* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
  #ifdef SIMPLE
    CHECKNULL(xFFT   =(MyFFTcompl   *)fftwf_malloc(sizeof(MyFFTcompl)  *Threads*fftSize));
    CHECKNULL(planFFT=(MyFFTCPUType *)fftwf_malloc(sizeof(MyFFTCPUType)*Threads)); 
  #else
    CHECKNULL(xFFT   =(MyFFTcompl   *) fftw_malloc(sizeof(MyFFTcompl)  *Threads*fftSize));
    CHECKNULL(planFFT=(MyFFTCPUType *) fftw_malloc(sizeof(MyFFTCPUType)*Threads)); 
  #endif
  for(k=0;k<Threads;k++)
  {
    #ifdef SIMPLE
      planFFT[k]=fftwf_plan_dft_1d(fftSize, &xFFT[k*fftSize], &xFFT[k*fftSize], FFTW_FORWARD, planType);
    #else
      planFFT[k]= fftw_plan_dft_1d(fftSize, &xFFT[k*fftSize], &xFFT[k*fftSize], FFTW_FORWARD, planType);
    #endif
  }

  #ifdef _OPENMP
    #pragma omp parallel num_threads(Threads)
  #endif
  {
    int myID, myIDpos, pos, i, j, diff1, diff2;

    #ifdef _OPENMP
      myID=omp_get_thread_num();
    #else
      myID=0;
    #endif
    myIDpos = myID*fftSize;
    diff1   = winSize-Overlap;
    diff2   = sizeof(MyFFTcompl)*(fftSize-winSize);

    #ifdef _OPENMP
      #pragma omp for
    #endif
    for(i=0; i<nFrames; i++)
    {
      pos=i*diff1;
      for(j=0; j<winSize; j++)
        xFFT[myIDpos+j]=audio[pos+j] * vHamming[j];

      if(winSize<fftSize)
        memset(&xFFT[myIDpos+winSize], 0, diff2);

      #ifdef SIMPLE
        fftwf_execute(planFFT[myID]);
      #else
         fftw_execute(planFFT[myID]);
      #endif

      pos=i*rowsNMF;
      for(j=0; j<rowsNMF; j++)
        #ifdef SIMPLE
          rSNMF[pos+j]=cabsf(xFFT[myIDpos+j]);
        #else
          rSNMF[pos+j]= cabs(xFFT[myIDpos+j]);
        #endif
    }
  }
  for(k=0;k<Threads;k++) {
    #ifdef SIMPLE
      fftwf_destroy_plan(planFFT[k]);
    #else
      fftw_destroy_plan(planFFT[k]);
    #endif
  }
  #ifdef SIMPLE
    fftwf_free(xFFT);
    fftwf_free(planFFT);
  #else
    fftw_free(xFFT);
    fftw_free(planFFT);
  #endif

  return OK;
}


int cONMF(const int F, const int T, const int K, const int K1, const MyType gamma, const MyType lamda, const int niter, MyType *X, MyType *W, MyType *H)
{
  int    i, K2=K-K1;
  MyType *Num=NULL, *Den=NULL, *tmp=NULL, *Klam=NULL, *Flam=NULL;

  // W needs auxiliary spaces of size (F, K2) and (K2, K2)
  // H needs auxiliary spaces of size (K, T)  and (K, K)
  // (K > K2) so we use (K, max(T,F)) and (K, K) for all
  CHECKNULL(Num =(MyType *)calloc(K*max(F,T), sizeof(MyType)));
  CHECKNULL(Den =(MyType *)calloc(K*max(F,T), sizeof(MyType)));
  CHECKNULL(tmp =(MyType *)calloc(K*K, sizeof(MyType)));

  CHECKNULL(Klam =(MyType *)calloc(K2, sizeof(MyType)));
  CHECKNULL(Flam =(MyType *)calloc(F,  sizeof(MyType)));


  /**********Normalization of X*********/
  if (gamma != (MyType)1.0) { cGamma(F*T, gamma, X); }

  /**********Fill Klam*********/
  for(i=0;i<K2;i++) { Klam[i]=lamda; }

  for(i=0;i<niter;i++)
  {
    #ifdef SIMPLE
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,   F,  K2, T,  1.0, X,        F, &H[K1], K,  0.0, Num, F);
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,   K2, K2, T,  1.0, &H[K1],   K, &H[K1], K,  0.0, tmp, K2);
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, F,  K2, K2, 1.0, &W[K1*F], F, tmp,    K2, 0.0, Den, F);

      cblas_sgemv(CblasColMajor, CblasNoTrans, F, K2, 1.0, &W[K1*F], F, Klam, 1, 0.0, Flam, 1);
    #else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,   F,  K2, T,  1.0, X,        F, &H[K1], K,  0.0, Num, F);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,   K2, K2, T,  1.0, &H[K1],   K, &H[K1], K,  0.0, tmp, K2);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, F,  K2, K2, 1.0, &W[K1*F], F, tmp,    K2, 0.0, Den, F);

      cblas_dgemv(CblasColMajor, CblasNoTrans, F, K2, 1.0, &W[K1*F], F, Klam, 1, 0.0, Flam, 1);
    #endif
    cUpdateL(F, K2, lamda, Num, Den, Flam, &W[K1*F]);

    #ifdef SIMPLE
      cblas_sgemm(CblasColMajor, CblasTrans,   CblasNoTrans, K, T, F, 1.0, W,   F, X, F, 0.0, Num, K);
      cblas_sgemm(CblasColMajor, CblasTrans,   CblasNoTrans, K, K, F, 1.0, W,   F, W, F, 0.0, tmp, K);
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, T, K, 1.0, tmp, K, H, K, 0.0, Den, K);
    #else
      cblas_dgemm(CblasColMajor, CblasTrans,   CblasNoTrans, K, T, F, 1.0, W,   F, X, F, 0.0, Num, K);
      cblas_dgemm(CblasColMajor, CblasTrans,   CblasNoTrans, K, K, F, 1.0, W,   F, W, F, 0.0, tmp, K);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, T, K, 1.0, tmp, K, H, K, 0.0, Den, K);
    #endif
    cUpdate(K*T, Num, Den, H);
  }
  free(Num); free(Den); free(tmp); free(Klam); free(Flam);
  return 0;
}


int ONMF(const MyType *audio, int sr, MyType S, int winSize, int Overlap, int nFrames, int rowsNMF, int fftSize,
          int bases, int freq, int nIter, MyType gamma, int Fmin, int Fmax, int K1, MyType *W, MyType *H, MyType *DFT,
          double *Tiempos, bool debug)
{
  int    planType=FFTW_ESTIMATE, Threads;
  MyType *rSNMF=NULL, *vHamming=NULL;
  double globaltime, dtime;

  #ifdef _OPENMP
    Threads = omp_get_max_threads();
  #else
    Threads = 1;
  #endif

  globaltime=Ctimer();

  if (debug) {
    printf("sr=%d, S=%1.2f, winSize=%d, Overlap=%d, nFrames=%d, rowsNMF=%d, fftSize=%d bases=%d, sr=%d, nIter=%d, gamma=%f fmin=%d, fmax=%d, K1=%d, Threads=%d\n", 
            sr, S, winSize, Overlap, nFrames, rowsNMF, fftSize, bases, sr, nIter, gamma, Fmin, Fmax, K1, Threads);
  }
  
  /* Internal structures */
  CHECKNULL(rSNMF    = (MyType *)malloc(rowsNMF*nFrames*sizeof(MyType)));
  CHECKNULL(vHamming = (MyType *)malloc(        winSize*sizeof(MyType)));

  /* Start !!! */
  cHamming(winSize, vHamming);
  dtime=Ctimer();
    cHammFFT(winSize, Overlap, nFrames, rowsNMF, fftSize, Threads, planType, audio, vHamming, rSNMF);
  Tiempos[0]=Ctimer()-dtime;

  dtime=Ctimer();
    cNorm(rowsNMF*nFrames, rSNMF);
  Tiempos[1]=Ctimer()-dtime;

  dtime=Ctimer();
    cBandPass(nFrames, rowsNMF, winSize, freq, Fmin, Fmax, rSNMF);
  Tiempos[2]=Ctimer()-dtime;

  dtime=Ctimer();
    cONMF(rowsNMF, nFrames, bases, K1, gamma, 0.2, nIter, rSNMF, W, H);
  Tiempos[3]=Ctimer()-dtime;

  dtime=Ctimer();
    cDFT(bases, K1, nFrames, Threads, planType, H, DFT);
  Tiempos[4]=Ctimer()-dtime;
  
  free(rSNMF); free(vHamming);

  Tiempos[5]=Ctimer()-globaltime;
  
  return 0;
}


void ctransp(const int r, const int c, const MyType *src, MyType *dest)
{
  int i, j;
  for(j=0;j<c;j++)
    for(i=0;i<r;i++)
      dest[i*c+j]=src[j*r+i]; // src is column-major
}


int cShiftFFT(const int K, const int T, MyType *dft_rows){
    int i, T2,K2, extra=0;
    T2=(int)floor(T/2.0);
    K2=(int)floor(K/2.0);
    MyType *temp, *line;
    if(T%2==1) extra=1;

    CHECKNULL(temp=(MyType *)malloc((T2+extra) * sizeof(MyType)));
    CHECKNULL(line=(MyType *)malloc((T) * sizeof(MyType)));

    if (K % 2==1) memcpy(line, &dft_rows[(K2)*T], T*sizeof(MyType));
    
    for(i=0;i<K2;i++)
    {
      memcpy(temp, &dft_rows[i*T], (T2+extra)*sizeof(MyType));
      memcpy(&dft_rows[i*T], &dft_rows[(i+K2+1)*T+T2+extra], T2*sizeof(MyType));
      memcpy(&dft_rows[(i+K2)*T+T2], temp, (T2+extra)*sizeof(MyType));
    	
      memcpy(temp, &dft_rows[i*T+T2+extra], T2*sizeof(MyType));
      memcpy(&dft_rows[i*T+T2], &dft_rows[(i+K2+1)*T], (T2+extra)*sizeof(MyType));
      memcpy(&dft_rows[(i+K2)*T], temp, T2*sizeof(MyType));
    }
    
    
    if (K % 2==1){
        memcpy(&dft_rows[(K-1)*T+T2],&line[0], (T2+extra)*sizeof(MyType));
        memcpy(&dft_rows[(K-1)*T],&line[T2+extra], T2*sizeof(MyType));      
    }
    
    free(temp); free(line);
    return 0;
}


int cSmooth(const int col, const int T, const int K, const int WSZ, MyType *H_soft, const MyType *H, MyType *temp)
{
  MyType dtmp;
  int j, lr, step;

  lr=WSZ/2; dtmp=0.0;

  for(j=0;j<WSZ;j++)
    dtmp += H[K*j+col];
  H_soft[lr] = dtmp / WSZ;

  step=lr+1;
  for(j=WSZ;j<T;j++) {
    dtmp += H[K*j+col] - H[K*(j-WSZ)+col];
    H_soft[step++] = dtmp / WSZ;
  }

  // Start
  dtmp=0.0;
  for(j=0;j<WSZ-1;j++) { dtmp += H[K*j+col]; temp[j] = dtmp; }
  for(j=0;j<lr;j++)    { H_soft[j] = temp[j*2] / (2*j+1); }

  // Stop
  dtmp=0.0; step=0;
  for(j=T-1;j>=T-WSZ+1;j--) { dtmp += H[K*j+col]; temp[step++] = dtmp; }

  // With soft_window=5 lr is 2
  for(j=1;j<lr;j++) { temp[j] = temp[j*2] / (2*j+1); }
  for(j=0;j<lr;j++) { H_soft[T-1-j] = temp[j];    }

  return 0;
}


int cDFT(const int K, const int K1, const int T, const int Threads, const int planType, const MyType *H, MyType *dft_cols)
{
  int          r, fftSize=T;
  MyType       *H_soft=NULL, *temp=NULL;
  MyFFTcompl   *xFFT=NULL;
  MyFFTCPUType *planFFT=NULL;

  #ifdef SIMPLE
    CHECKNULL(xFFT   =(MyFFTcompl   *)fftwf_malloc(sizeof(MyFFTcompl)  *Threads*fftSize));
    CHECKNULL(planFFT=(MyFFTCPUType *)fftwf_malloc(sizeof(MyFFTCPUType)*Threads)); 
  #else
    CHECKNULL(xFFT   =(MyFFTcompl   *) fftw_malloc(sizeof(MyFFTcompl)  *Threads*fftSize));
    CHECKNULL(planFFT=(MyFFTCPUType *) fftw_malloc(sizeof(MyFFTCPUType)*Threads)); 
  #endif
  for(r=0;r<Threads;r++)
  {
    #ifdef SIMPLE
      planFFT[r]=fftwf_plan_dft_1d(T, &xFFT[r*fftSize], &xFFT[r*fftSize], FFTW_FORWARD, planType);
    #else
      planFFT[r]= fftw_plan_dft_1d(T, &xFFT[r*fftSize], &xFFT[r*fftSize], FFTW_FORWARD, planType);
    #endif
  }
  CHECKNULL(H_soft=(MyType *)calloc(K1*T,                sizeof(MyType)));
  CHECKNULL(temp=  (MyType *)calloc(soft_window*Threads, sizeof(MyType)));
    
  #ifdef _OPENMP
    #pragma omp parallel num_threads(Threads)
  #endif
  {
    int    myID, posFFT, posH, i, j;
    MyType norm;

    #ifdef _OPENMP
      myID=omp_get_thread_num();
    #else
      myID=0;
    #endif
    posFFT = myID*fftSize;

    #ifdef _OPENMP
      #pragma omp for
    #endif
    for(i=0;i<K1;i++)
    {
      posH = i*T;
      cSmooth(i, T, K, soft_window, &H_soft[posH], H, &temp[myID*soft_window]);
      #ifdef SIMPLE
        norm=cblas_snrm2(T, &H_soft[posH], 1);
        cblas_sscal(T, (1.0 / norm), &H_soft[posH], 1);
      #else
        norm=cblas_dnrm2(T, &H_soft[posH], 1);
        cblas_dscal(T, (1.0 / norm), &H_soft[posH], 1);
      #endif

      for(j=0; j<T; j++) { xFFT[posFFT+j]=H_soft[posH+j]; }
      
      #ifdef SIMPLE
        fftwf_execute(planFFT[myID]);
      #else
         fftw_execute(planFFT[myID]);
      #endif

      for(j=0; j<T; j++)
      {
        #ifdef SIMPLE
          H_soft[posH+j]=cabsf(xFFT[posFFT+j]);
        #else
          H_soft[posH+j]= cabs(xFFT[posFFT+j]);
        #endif
      }
    }
  }
  cShiftFFT(K1, T, H_soft);
  ctransp(T, K1, H_soft, dft_cols);

  for(r=0;r<Threads;r++) 
  {
    #ifdef SIMPLE
      fftwf_destroy_plan(planFFT[r]);
    #else
      fftw_destroy_plan(planFFT[r]);
    #endif
  }
  #ifdef SIMPLE
    fftwf_free(xFFT);
    fftwf_free(planFFT);
  #else
    fftw_free(xFFT);
    fftw_free(planFFT);
  #endif
  free(temp); free(H_soft);

  return OK;
}
