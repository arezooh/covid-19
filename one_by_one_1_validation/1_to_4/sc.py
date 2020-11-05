from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom


def main():
    
    for i in range(4):
        print(i)
        subprocess.call("python ./prediction.py "+str(i), shell=True)


if __name__ == "__main__":

    main()
