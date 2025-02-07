import sys
from RRprocess import *

# Entry point function. The parameters are 
#   a) path audio files
#   b) number of repetitions for ONMF factorization
def main():
  args = sys.argv[1:]

  if len(args)==2:
    ficheros = NombresFicheros(args[0]) 

    for i in ficheros:
      fichero=args[0]+i
      RR=process(fichero, args[1])
      print(f"Estimated Respiratory Rate: {RR:2d} rpm para el fichero {fichero}")
  else:
    print("USE: RRmain.py <path audio files> <repetitions>")
    return

if __name__ == "__main__":
    main()
