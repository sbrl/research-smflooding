# From Nina

import sys, os
import shutil
from os import listdir
from PIL import Image



# remove empty files

path = "./img/"

c = 0

files = os.listdir(path)

for f in files:
	if os.stat(path+f).st_size == 0:
		os.remove(path+f)
		c = c + 1
#		shutil.move(path+f,"./empty/"+f) 
		print(f)
		
print(c)		

		
		


# remove corrupted files

c = 0

for filename in listdir('./img/'):
	if filename.endswith('.jpg'):
		try:
			img = Image.open('./img/'+filename) # open the image file
			img.verify() # verify that it is, in fact an image
		except (IOError, SyntaxError) as e:
			print('Bad file:', filename) # print out the names of corrupt files	
			os.remove('./img/'+filename)	
			c = c+1
			
print(c)			



# remove files that TF cannot read

import tensorflow as tf
from tensorflow import keras

path = "./img/"

files = os.listdir(path)

c = 0

for f in files:
	if f.endswith(".jpg"):
		print("fffffffff",f)
#		image1 = tf.io.decode_image(path+f)
		
		try:
			image = tf.keras.preprocessing.image.load_img(path+f)
			input_arr = keras.preprocessing.image.img_to_array(image)
		except:
			os.remove(path+f)			
			c = c+1
			
print(c)			
