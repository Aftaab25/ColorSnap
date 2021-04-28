#importing libraries
import cv2 # opencv to read and show the image and display results
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from colormap import rgb2hex # to convert (r, g, b) values to hex values
#import argparse

#Creating an argument parser to take image from the command line
#argp = argparse.ArgumentParser()
#argp.add_argument('-i', '--image', required=True, help="Image Path")
#args = vars(argp.parse_args())
#img_path = args['image']

# Reading the image with opencv
img_path = '/home/aftaab/Projects/ColorZilla_Clone/tree.jpg' # Image path of the image to be used
in_img = cv2.imread(img_path) #reading image from the given path
img = cv2.resize(in_img, (1000, 800))


clicked = False # global variable
r = g = b = xpos = ypos = 0 # global variables

name = ["color", "color_name", "hex", "R", "G", "B"]
dataset = pd.read_csv('colors.csv', names=name, header=None)
# x = dataset.iloc[:, [3,5]].values
# 
# kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
# y_means = kmeans.fit_predict(x)
# 
# plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s=100, c='green', label='Cluster 3')
# 
# plt.title('The Output')
# plt.xlabel('Colors')
# plt.ylabel('values')
# plt.legend()
# plt.show()

# Calculate the minimum distance from each color to find the closest matching color
def getColor(R, G, B):
    minimum = 10000 # initialising minimum with random value
    for i in range(len(dataset)):
        dist = abs(R - int(dataset.loc[i, "R"])) + abs(G - int(dataset.loc[i, "G"])) + abs(B - int(dataset.loc[i, "B"]))
        if(dist<=minimum):
            minimum = dist
            color_found = dataset.loc[i, "color_name"] # finding the color that corresponds to the found distance
    #return color_found
    #print(minimum)
    return (color_found, minimum)    

"""
To find the color of the given point, we first need to detect the (x, y) coordinates of the point clicked
This can be done with the help of cv2.EVENT_LBUTTONDBLCLK 
Function Argumments:
@event => corresponds to the events triggered by the mouse
@x, @y => Global variables
@flags => corresponds to the type of image read by cv2.imread()
@param => To add additional arguments whenever needed
""" 

def get_pos(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

cv2.namedWindow('image')
cv2.setMouseCallback('image', get_pos)

while(1): # infinite loop
    cv2.imshow("image", img)
    if(clicked):
        """
        rectangle takes multiple arguments
        """
        #cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
        #text = getColor(r, g, b) + " R=" + str(r) + " G="+ str(g) + " B=" + str(b) + " " + str(rgb2hex(r, g, b))
        #cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        #if(r + g + b >= 600):
        #    cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        #clicked = False 

        cv2.rectangle(img, (200, 20), (880, 60), (b, g, r), -1)
        #cv2.rectangle(img, (10, 60), (900, 100), (b, g, r), -1)
        #cv2.rectangle(img, (10, 100), (900, 140), (b, g, r), -1)
        cf, accuracy = getColor(r, g, b)
        accuracy = (765 - accuracy) / 765 
        accuracy = accuracy*100
        acc = "{:.2f}".format(accuracy)
        #print(acc)
        text1 = cf + " R=" + str(r) + " G="+ str(g) + " B=" + str(b) + " " + str(rgb2hex(r, g, b)) + " " + acc + "%"
        if(r + g + b >= 500):
            cv2.putText(img, text1, (210, 45), 2, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, text1, (210, 45), 2, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


        clicked = False

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
