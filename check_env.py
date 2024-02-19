import os
if __name__=="__main__":
    if os.getenv('virenv'):
        print("Conda environment is active")
    else:
        print("Conda environment is not active")