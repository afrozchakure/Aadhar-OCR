
# Import packages
import dateutil.parser as dparser
import csv
import re
import pytesseract
import json
import os.path
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'
IMAGE_NAME = 'test_images/image1.jpg'

image_path = 'test_images/image1.jpg'  # Giving the input image
# Givin the path to output image
output_path = 'test_images/output/output.jpg'
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=3,
    min_score_thresh=0.60)

ymin, xmin, ymax, xmax = array_coord

shape = np.shape(image)
im_width, im_height = shape[1], shape[0]
(left, right, top, bottom) = (xmin * im_width,
                              xmax * im_width, ymin * im_height, ymax * im_height)

# Using Image to crop and save the extracted copied image
im = Image.open(image_path)
im.crop((left, top, right, bottom)).save(output_path, quality=95)

cv2.imshow('ID-CARD-DETECTOR : ', image)

image_cropped = cv2.imread(output_path)
cv2.imshow("ID-CARD-CROPPED : ", image_cropped)

# All the results have been drawn on image. Now display the image.
cv2.imshow('ID CARD DETECTOR', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()

#-----------------------Code to Extract Information from the Image-----------------------#

# Module to extract information from the image
# path = sys.argv[1] # To get the path from the user

img = Image.open(output_path)
img = img.convert('RGBA')
pix = img.load()

for y in range(img.size[1]):
    for x in range(img.size[0]):
        if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
            pix[x, y] = (0, 0, 0, 255)
        else:
            pix[x, y] = (255, 255, 255, 255)

img.save('temp.png')

text = pytesseract.image_to_string(Image.open('temp.png'))

# Initializing data variable
name = None
gender = None
ayear = None
uid = None
yearline = []
genline = []
nameline = []
text1 = []
text2 = []
genderStr = '(Female|Male|emale|male|ale|FEMALE|MALE|EMALE)$'


# Searching for Year of Birth
lines = text
# print (lines)
for wordlist in lines.split('\n'):
    xx = wordlist.split()
    if [w for w in xx if re.search('(Year|Birth|irth|YoB|YOB:|DOB:|DOB)$', w)]:
        yearline = wordlist
        break
    else:
        text1.append(wordlist)
try:
    text2 = text.split(yearline, 1)[1]
except Exception:
    pass

try:
    yearline = re.split('Year|Birth|irth|YoB|YOB:|DOB:|DOB', yearline)[1:]
    yearline = ''.join(str(e) for e in yearline)
    if yearline:
        ayear = dparser.parse(yearline, fuzzy=True).year
except Exception:
    pass

# Searching for Gender
try:
    for wordlist in lines.split('\n'):
        xx = wordlist.split()
        if [w for w in xx if re.search(genderStr, w)]:
            genline = wordlist
            break
#        print("wordlist:", xx)  # To print the word list
#    print(genline)  # To print the genlines
    if 'Female' in genline or 'FEMALE' in genline:
        gender = "Female"
    elif 'Male' in genline or 'MALE' in genline:
        gender = "Male"

    text2 = text.split(genline, 1)[1]
except Exception:
    pass

# Read Database
with open('namedb1.csv', 'r') as f:
    reader = csv.reader(f)
    newlist = list(reader)
newlist = sum(newlist, [])

# Printing the name of the user
print("Print Name: ", text1[1])
name_new = text1[1]  # Extracting name from the image


"""
# Searching for Name and finding exact name in database
try:
    text1 = filter(None, text1)
    for x in text1:
        for y in x.split():
            if y.upper() in newlist:    
                nameline.append(x)
                print(x)
                break
    name = ' '.join(str(e) for e in nameline)
except Exception:
    pass
"""

# Searching for UID
uid = []  # Empty list for uid
try:
    newlist = []
    for xx in text2.split('\n'):
        newlist.append(xx)
    newlist = list(filter(lambda x: len(x) > 12, newlist))
    # This extracts the vid from the image
    print("Extracted UID:", newlist[2][6:])
    uid_new = newlist[2][6:]  # New vid from the image
    for no in uid_new:
        #        print("Uid: ", no)  # To print individual uid elements
        if re.match("^[0-9 ]+$", no):
            uid.append(no)

except Exception:
    pass

# Making tuples of data
data = {}
data['Name'] = name_new
data['Gender'] = gender
data['Birth year'] = ayear
if len(list(uid)) > 1:
    data['Uid'] = "".join(uid)  # Storing uid into data['Uid']
else:
    data['Uid'] = None

# Writing data into JSON
fName = 'result/' + os.path.basename(output_path).split('.')[0] + '.json'
with open(fName, 'w') as fp:
    json.dump(data, fp)

# Removing dummy files
os.remove('temp.png')

# Reading data back JSON
with open(fName, 'r') as f:
    ndata = json.load(f)


print("\n-----------------------------\n")
print(ndata['Name'])
print("-------------------------------")
print(ndata['Gender'])
print("-------------------------------")
print(ndata['Birth year'])
print("-------------------------------")
print(ndata['Uid'])
print("-------------------------------")
