import os
import subprocess
import random
import string

command = "rm groups_dataset.csv"
process = subprocess.Popen(command, shell=True)
process.wait()

#generate some synthetic texts
msfontdir = "/usr/share/fonts/truetype/msttcorefonts/"
#dictionery with words examples
words = ["llnear","comblnatlon","any","system","vectors","wlth","all","zero","coefflclents","zero","vector","only","way","express","zero","vector","llnear","comblnatlon","these","vectors","are","llnearly","lndependent","Glven","set","vectors","that","span","space,","any","vector","llnear","comblnatlon","other","vectors","set","not","llnearly","lndependent","then","span","would","remaln","the","same","remove","from","the","set.","Thus,","set","llnearly","dependent","vectors","redundant","the","sense","that","llnearly","lndependent","subset","wlll","span","same","subspace.","Therefore,","are","mostly","lnterested","llnearly","lndependent","set","vectors","that","spans","vector","space","whlch","call","basls","","Any","set","vectors","that","spans","contalns","basls,","and","any","llnearly","lndependent","set","vectors","can","extended","basls","turns","out","that","accept","axlom","cholce,","every","vector","space","has","basls","nevertheless,","thls","basls","may","unnatural,","and","lndeed","may","not","even","constructable.","For","lnstance","there","exlsts","basls","for","real","numbers","consldered","vector","space","over","the","ratlonals,","but","expllclt","basls","has","been","constructed"]

#generate some synthetic characters from MS core fonts
fonts = ["Andale_Mono.ttf","andalemo.ttf","arialbd.ttf","arialbi.ttf","Arial_Black.ttf","Arial_Bold_Italic.ttf","Arial_Bold.ttf","Arial_Italic.ttf","ariali.ttf","arial.ttf","Arial.ttf","ariblk.ttf","comicbd.ttf","Comic_Sans_MS_Bold.ttf","Comic_Sans_MS.ttf","comic.ttf","courbd.ttf","courbi.ttf","Courier_New_Bold_Italic.ttf","Courier_New_Bold.ttf","Courier_New_Italic.ttf","Courier_New.ttf","couri.ttf","cour.ttf","Georgia_Bold_Italic.ttf","Georgia_Bold.ttf","georgiab.ttf","Georgia_Italic.ttf","georgiai.ttf","georgia.ttf","Georgia.ttf","georgiaz.ttf","impact.ttf","Impact.ttf","timesbd.ttf","timesbi.ttf","timesi.ttf","Times_New_Roman_Bold_Italic.ttf","Times_New_Roman_Bold.ttf","Times_New_Roman_Italic.ttf","Times_New_Roman.ttf","times.ttf","trebucbd.ttf","trebucbi.ttf","Trebuchet_MS_Bold_Italic.ttf","Trebuchet_MS_Bold.ttf","Trebuchet_MS_Italic.ttf","Trebuchet_MS.ttf","trebucit.ttf","trebuc.ttf","Verdana_Bold_Italic.ttf","Verdana_Bold.ttf","verdanab.ttf","Verdana_Italic.ttf","verdanai.ttf","verdana.ttf","Verdana.ttf","verdanaz.ttf"];
for i in range(0,731):
  num_words = random.randrange(1,5)
  text = ""
  for j in range(0,num_words):
    separator = " "
    if (random.randrange(0,10) > 5):
      separator = "\\n";
    if (random.randrange(0,10) > 5):
      text = text + words[random.randrange(0,len(words))] + separator
    else:
      text = text + words[random.randrange(0,len(words))].upper() + separator

    
  gen_command = "convert -background white -fill black -font "+msfontdir+fonts[i%len(fonts)]+" -pointsize "+str(random.randrange(100, 198))+" label:'"+text+"' -rotate "+str(random.randrange(0,360))+" -page +0+0 synth.tiff"
#convert -background white -fill black -font Cursi -pointsize 24 -gravity center label:'ImageMagick\n'  label_centered.gif
  process = subprocess.Popen(gen_command, shell=True)
  process.wait()
  print gen_command
  command = "./extract_group_features synth.tiff C >> groups_dataset.csv"
  gen_process = subprocess.Popen(command, shell=True)
  gen_process.wait()

# labeled boundaries
for dirname, dirnames, filenames in os.walk('../../Escriptori/text_extraction/data/train/groups/TEXT/'):
  for filename in filenames:
    if ('jpg' in filename):
        image_filename = os.path.join(dirname, filename)
        print image_filename
        command = "./extract_group_features "+image_filename+" C >> groups_dataset.csv"
        process = subprocess.Popen(command, shell=True)
        process.wait()

for dirname, dirnames, filenames in os.walk('../../Escriptori/text_extraction/data/train/groups/NO_TEXT/'):
  for filename in filenames:
    if ('jpg' in filename):
        image_filename = os.path.join(dirname, filename)
        print image_filename
        command = "./extract_group_features "+image_filename+" N >> groups_dataset.csv"
        process = subprocess.Popen(command, shell=True)
        process.wait()
