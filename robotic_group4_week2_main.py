import numpy as np
import cv2 as cv
import imutils

# Intesection over Union
def iOU(box1, box2):
    # Boxes' coordinate & size
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    # Get the width & height of the Intersection's area 
    intersection_width = min(x1 + width1, x2 + width2) - max(x1, x2)
    intersection_height = min(y1 + height1, y2 + height2) - max(y1, y2)

    # no overlap
    if width_intersection <= 0 or height_intersection <= 0: 
        return 0
    
    area_intersection = width_intersection * height_intersection
    area_union = width1 * height1 + width2 * height2 - area_intersection

    #Area IoU
    return I / U


vid_cap = cv.VideoCapture(0)

objectBox = ''
checkBox = (480, 230, 150, 250)
font = cv.FONT_HERSHEY_DUPLEX
fontScale = 0.5
thickness = 1
color = (0, 0, 255) 
org = (480, 220) 

# read the template 
template = cv.imread('./template/sample.jpg')
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY) 
template = cv.Canny(template, 50, 200)
   
(template_height, template_width) = template.shape[:2]

  

while True:
    _, frame = vidCap.read()
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.rectangle(frame, (checkBox[0], checkBox[1]), (checkBox[0] + checkBox[2], checkBox[1] + checkBox[3]), (255, 0, 0), 3)

    found = None
    for scale in np.linspace(0.3, 1, 3)[::-1]: 
        # Resize the image to the scale, keep track 
        # of the ratio of the resizing 
        resizedImg = imutils.resize(grayImg, width = int(grayImg.shape[1] * scale)) 
        scaleRatio = grayImg.shape[1] / float(resizedImg.shape[1]) 

        # If the resized image is smaller than the template, break
        if resizedImg.shape[0] < templateHeight or resizedImg.shape[1] < templateWidth: 
            break
     
        # Detect edges in the resized image, grayscale image and apply template  
        # matching to find the template in the video 
        edgeReImg  = cv.Canny(resizedImg, 50, 200) 
        result = cv.matchTemplate(edgeReImg, template, cv.TM_CCOEFF) 
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result) 
        # If a new maximum correlation value is found, then update 
        # the found variable if found is None or maxVal > found[0]: 
        found = (maxVal, maxLoc, scaleRatio) 

    (_, maxLoc, scaleRatio) = found
    (startX, startY) = (int(maxLoc[0] * scaleRatio), int(maxLoc[1] * r)) 
    (endX, endY) = (int((maxLoc[0] + templateWidth) * scaleRatio), int((maxLoc[1] + templateHeight) * scaleRatio)) 
    
    objectBox = (startX, startY, endX - startX, endY - startY)

    # Draw a bounding box for the result and display the image 
    cv.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2) 

   
    if objectBox != '':
        if(iOU(objectBox, checkBox) > 0.3):
            cv.putText(frame, 'detected', org, font, fontScale, color, thickness, cv.LINE_AA) 

    cv.imshow("frame", frame)

    key = cv.waitKey(1)
    if key == 27:
        break
vidCap.release()
cv.destroyAllWindows()
