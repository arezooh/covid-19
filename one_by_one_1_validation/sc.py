from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom


def main():
<<<<<<< HEAD:one_by_one_1_validation/1_to_4/sc.py
    
    for i in range(4):
=======
    loom = ProcessLoom(max_runner_cap = 8)
    for i in range(7):
>>>>>>> 5c60d8fa91da4126bd59c3e41b8253582f06fff2:one_by_one_1_validation/sc.py
        print(i)
        subprocess.call("python ./prediction.py "+str(i), shell=True)


if __name__ == "__main__":

    main()
