# https://youtu.be/3RNPJbUHZKs
"""
Remove text from images
"""

import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np

#General Approach.....
#Use keras OCR to detect text, define a mask around the text, and inpaint the
#masked regions to remove the text.
#To apply the mask we need to provide the coordinates of the starting and 
#the ending points of the line, and the thickness of the line

#The start point will be the mid-point between the top-left corner and 
#the bottom-left corner of the box. 
#the end point will be the mid-point between the top-right corner and the bottom-right corner.
#The following function does exactly that.
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

#Main function that detects text and inpaints. 
#Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img_path, pipeline):
    # read the image 
    img = keras_ocr.tools.read(img_path) 
    
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([img])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    inpainted_img = False
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

from pathlib import Path
import os
import glob
import requests
import shutil

inputFiles = glob.glob('1.input/*')
if len(inputFiles) == 0:
  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEjNFURpuYCoogk0ps9lW31Ns5xSR-c6PK8cWZIIYumBwO_JOcgLOMHq27SAZKWv0fhS4NmU4052XsnjMwn2LrVEqBVyChlmHKQ2K8Wi-hOLbE9AUsAOvgV29tqBa0etbV_xzu4nCAOvzoqpAHaFkgBAtBNF5BBc5A64N7jHKIVUH8pHPslFpXA"
  response = requests.get(URL)
  open("1.input/1-white-text.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEh3BMIRTSZPOq4W4bfLxxtXk2AA_JM-J-vj7PNLAl3iykXy3ZykRaE0l0iyEtbdT6l20rIAqFskP-4poB0QNbSCxCptXNlqHhuO62GNeYBou389lH0VWaZUEJ_kXDiPhO0am0vxjqsFkcPheXWinvQyQjpsQvTnz9iIZYzztqe-XDZjjuDoloQ"
  response = requests.get(URL)
  open("1.input/2-black-text.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEjgtOkL9tAb3SZnidC7MlfSsOBRpXUg3Jj5LfjTQE42C_OtmWfxrfKVKCBeB_44GlwPHc4J3m4liVtsC7eJUs6QsscflPsSpQPDizJSHaYwe-XYhRH81KbXiKnNpDJxfIdwbpbTJK-KcrhmBrXakKOJA_jWpS2EkrJi1B-uLVkhRuTJHAmemhI"
  response = requests.get(URL)
  open("1.input/3-no-text.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEj8kNtxp7SK1ewRouTXlpRxMCjrQLtuyPwmuatiy4YFvDNdlyZrZqFWcZfjBM2T-zf6ApckoMJ6I1mn-mq32beLQqLi5iVtILvMBCOGHH1NDuQKqJ28LpaWtjA3ajKtT7mkll8Pr1lcMM8z3CFXC-w_NuAWW6o-GoAgoyWgX3--eNvWEPDlLWI"
  response = requests.get(URL)
  open("1.input/4-1-girl-cat.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEhgdZWiqkA5b6mT5uxWab_i7X2ExN7Mwr7-ElcNsBwdiZ7kjYfgpZqAjevnbdNajKEaNw8glcizAnmJvCKzGFs55xiQoeO-Ed5amIh3D_ztmTVT9Rc7HXj8QFOvgPG7voinCtcQtJpbfrklFvzk2FBa8JyNqm4N8Exb4pSFjmgDDsn6e2480iE"
  response = requests.get(URL)
  open("1.input/4-2-girl-cat-small.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEjyH7H0BtHn1i0nJJjeqHB6jTSw9DpRONqwvRhtsaw0MWOzxE7SWeuX2r4NerzWyoCQY1FMTwwt6qIc9uJPIOUgHi2sg1lNEc0snIk9BI2TpKxvM58vjcxuXGlqogv4PfUOgLEPd4S5UP8bjIAgVZ-VrPLg2CafHMIDTC-BcygAyV2Tm5nkYH4"
  response = requests.get(URL)
  open("1.input/4-3-girl-cat.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEjRprnLvf97BAU7kpk_sfZn0vXBzkSn--n8RdkorC6OZoPLln80SyOcDVh4Vl7Za-ELu88kqpBYqTBKnDJLmiSpFtENshnjZuYbeXYf4jty1M6K_vmlDf_aehdXa9Eb1mcnrCIFmMZKnJVjVDeS-je2jQD_3E0fup4oQpO3ThLKQ7jFC8A2Yus"
  response = requests.get(URL)
  open("1.input/5-1-comic.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEi5P_dfvD2mY4aGjQy5AEKwCEWIto6AuRTnelOWxKW_gA0LFfeg2eqpdwYiqSXmKXIZkP_zj7fG4LlIhgTuqGzdSOC3YXj3emwBBmLgY3pFv4Is3E56x1uyhwb-50sFsUVv9VDpv-Accc7m3f9nmao5Y6QYy6eiel6i30S1TUrHbm2Av740I1w"
  response = requests.get(URL)
  open("1.input/5-2-comic.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEhWCzc50eZPJJKNR_zpR1hi7XltJHNsF-YdTuJ10T4Y1uJF8j1yfHfWyqfEdLMevTIgVFYQtGFW9IxKaoXtLY9UkTzG_Xd1ZiqtSSDsOchcCv9G6Ovit5K-3FCmJsUhXRMmYIcxekpOigV75W-5SHxzK1n-XpQW09Nyip7ia8mY5VevQDwHji4"
  response = requests.get(URL)
  open("1.input/6-1-no-face-seal.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEjcoyA9IheRnWPXeUzDzehfCNzhMn89o_4vQ7Afwe7p7SGKm3MMC97bwTNv86NTHWvcHE9DLZGJteq4TMgSAtZw5FsNJSe6h3gBZHh7fAqy5kSfh87r6cEck0lBn7lp8FhI5PycSsu80iciZQX2VXGMXlFLLYFOQ65nFNPescdyX3SnCo-eW9M"
  response = requests.get(URL)
  open("1.input/6-2-no-face-glass.jpg", "wb").write(response.content)

  URL = "https://blogger.googleusercontent.com/img/a/AVvXsEhvHAL06WwuIL52PV5UIV9pzKLlzdpGFBOzuqFQVhX5WUVDUT1vwhRN3RxuOtkpLrBmUZ_AHubRFzfa0WRRevsTeoEUZI0SVQPizs0_PvwEy3cqvzi5Svj2GAWBICxHYe1AMU5yNBBaHu5oM_qQW3Ak8DGK1JCpEBPk919RHAthJV2bwMKjEV4"
  response = requests.get(URL)
  open("1.input/6-3-no-face-beach.jpg", "wb").write(response.content)
  
from pathlib import Path
import os
import glob
import requests
import shutil

inputFiles = glob.glob('1.input/*')


for inputFile in inputFiles:
  p = Path(inputFile)
  name = p.name
  outputFile = "2.inpaint/" + name
  
  if os.path.isfile(outputFile):
    print('File is exsited: ' + name)
    continue

  print('Processing: ' + name)

  img_text_removed = inpaint_text(inputFile, pipeline)
  if img_text_removed is False:
    shutil.copyfile(inputFile, outputFile)
  else:
    cv2.imwrite(outputFile, cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))

from pathlib import Path
import os
import glob
import requests
import shutil
from PIL import Image
from autocrop import Cropper
import subprocess

cropper = Cropper(face_percent=70)

inputFiles = glob.glob('2.inpaint/*')
for inputFile in inputFiles:
  p = Path(inputFile)
  name = p.name
  outputFile = "3.output/" + name
  
  if os.path.isfile(outputFile):
    print('File is exsited: ' + name)
    # continue

  print('Processing: ' + name)

  image = cv2.imread(inputFile)
  height, width, channels = image.shape
  if width < 300:
    image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('/tmp/tmp.jpg', image)
    inputFile = '/tmp/tmp.jpg'
  
  cropped_array = cropper.crop(inputFile)

  # print(cropped_array)
  # Save the cropped image with PIL if a face was detected:
  if cropped_array is None:
    #shutil.copyfile(inputFile, outputFile)
    subprocess.run(["smartcroppy","--width","300","--height","300",inputFile,outputFile])
  else:
    # cropped_image = Image.fromarray(cropped_array)
    #cropped_image.save(outputFile)
    cv2.imwrite(outputFile, cv2.cvtColor(cropped_array, cv2.COLOR_BGR2RGB))