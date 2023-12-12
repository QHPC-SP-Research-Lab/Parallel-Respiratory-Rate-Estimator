CC    := gcc
FLAGS := -Wall -O3 -fPIC -shared -fopenmp

# For BLIS
INCS  := -I./ -I/opt/fftw/include -I/opt/blis/include
LIBS  := -L/opt/blis/lib -lblis -L/opt/fftw/lib -lfftw3_omp -lfftw3

# For OpenBLAS
INCS  := -I./ -I/opt/fftw/include -I/opt/openblas/include
LIBS  := -L/opt/openblas/lib -lopenblas -L/opt/fftw/lib -lfftw3_omp -lfftw3

all: cleanall RRmain.so

cleanall: clean
	@rm -rf RRmain.so __pycache__

clean:
	@rm -f *.o core *~ 
	
RRmain.so: RRfunctions.c
	$(CC) -o RRmain.so RRfunctions.c $(FLAGS) $(INCS) $(LIBS)
