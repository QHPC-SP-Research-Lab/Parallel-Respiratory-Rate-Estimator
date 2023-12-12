import sys
from RRprocess import *

# Entry point function. The parameters are 
#   a) audio fullname (with the '.wav' extension)
#   b) number of repetitions for ONMF factorization
def main():
  args = sys.argv[1:]

  if len(args)==2:
    RR=process(args[0], args[1])
  else:
    print("USE: RRmain.py <file_name> <repetitions>")
    return
  print(f"Estimated Respiratory Rate: {RR} rpm")

if __name__ == "__main__":
    main()
