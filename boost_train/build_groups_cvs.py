import os
import subprocess
import random
import string

command = "rm groups_dataset.csv"
process = subprocess.Popen(command, shell=True)
process.wait()

#generate some synthetic characters

# labeled boundaries
for dirname, dirnames, filenames in os.walk('../../data/train/groups/TEXT/'):
	for filename in filenames:
	    if ('jpg' in filename):
		image_filename = os.path.join(dirname, filename)
		print image_filename
    		command = "./extract_group_features "+image_filename+" C >> groups_dataset.csv"
    		process = subprocess.Popen(command, shell=True)
    		process.wait()

for dirname, dirnames, filenames in os.walk('../../data/train/groups/NO_TEXT/'):
	for filename in filenames:
	    if ('jpg' in filename):
		image_filename = os.path.join(dirname, filename)
		print image_filename
    		command = "./extract_group_features "+image_filename+" N >> groups_dataset.csv"
    		process = subprocess.Popen(command, shell=True)
    		process.wait()
