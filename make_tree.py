import os
cwd = os.getcwd()
os.makedirs(os.path.join(cwd,'vel_data'),exist_ok=True)
os.makedirs(os.path.join(cwd,'results'), exist_ok=True)
os.makedirs(os.path.join(cwd,'errors'), exist_ok=True)
