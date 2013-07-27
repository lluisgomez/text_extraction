import os
import subprocess
import random
import string


command = "rm char_dataset.csv"
process = subprocess.Popen(command, shell=True)
process.wait()

#generate some synthetic characters
fonts = ["verdana.ttf","arial.ttf","times.ttf","comic.ttf"]
for i in range(0,7500):
	gen_command = "convert -background black -fill white -font /usr/share/fonts/truetype/msttcorefonts/"+fonts[i%4]+" -pointsize "+str(random.randrange(18, 98))+" label:"+random.choice(string.ascii_letters)+" -rotate "+str(random.randrange(0,360))+" -page +0+0 synth.tiff"
	process = subprocess.Popen(gen_command, shell=True)
	process.wait()
	print gen_command
	command = "./extract_char_features synth.tiff C >> char_dataset.csv"
	gen_process = subprocess.Popen(command, shell=True)
	gen_process.wait()


# labeled boundaries
for dirname, dirnames, filenames in os.walk('../../Escriptori/text_extraction/data/train/characters/CHARS'):
	for filename in filenames:
	    if ('jpg' in filename):
		image_filename = os.path.join(dirname, filename)
		print image_filename
    		command = "./extract_char_features "+image_filename+" C >> char_dataset.csv"
    		process = subprocess.Popen(command, shell=True)
    		process.wait()

for dirname, dirnames, filenames in os.walk('../../Escriptori/text_extraction/data/train/characters/NO_CHARS'):
	for filename in filenames:
	    if ('jpg' in filename):
		image_filename = os.path.join(dirname, filename)
		print image_filename
    		command = "./extract_char_features "+image_filename+" N >> char_dataset.csv"
    		process = subprocess.Popen(command, shell=True)
    		process.wait()
