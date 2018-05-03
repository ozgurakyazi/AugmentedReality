import numpy as np
import cv2 as cv
#from matplotlib import pyplot as plt

def draw_rect(contours,img):
    all_res = []
    for c in contours:
        result = cv.approxPolyDP(c, cv.arcLength(c,True)*0.02, closed=True)
        # print("start")
        # print(result)
        # print("done")

        # print("start")
        #print(img)
        # print("end")
        if result.size ==8:
            x,y,w,h = cv.boundingRect(result)
            if h<20 or w<20 or w > img.shape[1] - 10 or h > img.shape[0] - 10:
                pass
            else:
                cv.polylines(img, [result], 1, (0,0,255),4)
                all_res.append(np.int32(result))
    return all_res




def draw_min_rect(countours,img):
    for cnt in countours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img,[box],0,(0,0,255),2)

def draw_surrounding(countours,img):
    for cnt in countours:
        result = cv.approxPolyDP(cnt, cv.arcLength(cnt,True)*0.02, closed=True)

        if result.size ==8:
            cv.polylines(img, np.int32(result), 1, (0,0,255))

def draw_corner(results, img,num_mid_point=7 ):
    for res in results:
        for i,point in enumerate(res):
            next_point = res[(i+1)%len(res)]
            cv.circle(img, tuple(point[0]), 3, (0, 255, 0), -1) # draw the corner
            start_x = point[0][0]
            start_y = point[0][1]

            diff_x = int((point[0][0] - next_point[0][0]) / num_mid_point)
            diff_y = int((point[0][1] - next_point[0][1]) / num_mid_point)

            for middle_dot in range(num_mid_point):
                temp_x = start_x - diff_x*(middle_dot+1)
                temp_y = start_y - diff_y*(middle_dot+1)
                cv.circle(img, (temp_x,temp_y), 3, (255, 0, 0), -1) # draw the inbetween


cap = cv.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray =  cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 150,255,0)
    #th1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)

    # find contours
    im2, contours, hier = cv.findContours(thresh,1,cv.CHAIN_APPROX_SIMPLE)

    approx_conts = []
    #draw_min_rect(contours,frame)
    all_res = draw_rect(contours,frame)

    draw_corner(all_res,frame)
    #draw_surrounding(contours,frame)

    #cv.drawContours(frame2,contours,-1,(0,0,255),2)
    #cv.drawContours(frame,approx_conts,-1,(0,0,255),2)
    cv.imshow('simple_threshold',thresh)
    cv.imshow('nr',frame)
    #cv.imshow('nr2',frame2)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
