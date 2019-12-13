import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)

origin_x=260
origin_y=190
region_size = 110

flag_colour = 127
result_region_size = 2
result_region_length = 35
set_period = 2.5



kernel = np.ones((5,5), np.float32)/25
grabbed,frame = camera.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
imshow = frame
mark = np.full(frame.shape,0,dtype=np.uint8)
mark[ origin_y:origin_y+region_size,origin_x:origin_x+region_size] = 1

cot = region_size//2
judgment_area_x2_min = origin_x+cot-result_region_size
judgment_area_x2_max = origin_x+cot+result_region_size
judgment_area_y2_min = origin_y+cot-result_region_size
judgment_area_y2_max = origin_y+cot+result_region_size

judgment_area_x1_min = judgment_area_x2_min 
judgment_area_x1_max = judgment_area_x2_max
judgment_area_y1_min = judgment_area_y2_min - result_region_length
judgment_area_y1_max = judgment_area_y2_max - result_region_length

judgment_area_x3_min = judgment_area_x2_min
judgment_area_x3_max = judgment_area_x2_max
judgment_area_y3_min = judgment_area_y2_min + result_region_length
judgment_area_y3_max = judgment_area_y2_max + result_region_length

judgment_area_x_min_list = [judgment_area_x1_min, judgment_area_x2_min, judgment_area_x3_min]
judgment_area_x_max_list = [judgment_area_x1_max, judgment_area_x2_max, judgment_area_x3_max]
judgment_area_y_min_list = [judgment_area_y1_min, judgment_area_y2_min, judgment_area_y3_min]
judgment_area_y_max_list = [judgment_area_y1_max, judgment_area_y2_max, judgment_area_y3_max]
flag_point_list = [1,1,1]

flag3_buff = 0
time_vl = 0.0

success = 0
failure = 0
state_machine = 0

while True:
    grabbed,frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    image_original = frame*mark
    image = cv2.filter2D(image_original[:,:],-1,kernel)
    ret, image = cv2.threshold(image, 165, flag_colour, cv2.THRESH_BINARY)
    ##image = cv2.Canny(image, 280, 380)
    
    imshow = image
    imshow[0:region_size,0:region_size] = image_original[ origin_y:origin_y+region_size,origin_x:origin_x+region_size]
    
    flag1 = np.median(image[judgment_area_y_min_list[0]:judgment_area_y_max_list[0],judgment_area_x_min_list[0]:judgment_area_x_max_list[0]])
    flag2 = np.median(image[judgment_area_y_min_list[1]:judgment_area_y_max_list[1],judgment_area_x_min_list[1]:judgment_area_x_max_list[1]])
    flag3 = np.median(image[judgment_area_y_min_list[2]:judgment_area_y_max_list[2],judgment_area_x_min_list[2]:judgment_area_x_max_list[2]])
    
    if( flag3!=flag_colour and flag3_buff==flag_colour):
        time_vl = time.time()
    if( flag3==flag_colour and flag3_buff!=flag_colour):
        vl = time.time() - time_vl
        if(vl<set_period+0.5 and vl>set_period-0.5):
            success+=1
        else:
            failure+=1
        print(success,failure,'---',vl)

    flag3_buff = flag3

    flag_point_list = [0]*3
    if( flag1 == flag_colour):
        flag_point_list[0] = -1
    if( flag2 == flag_colour):
        flag_point_list[1] = -1
    if( flag3 == flag_colour):
        flag_point_list[2] = -1

    cv2.rectangle(imshow,  (origin_x,origin_y),(origin_x+region_size,origin_y+region_size),(255, 255, 255), 1)
    cv2.rectangle(imshow,  (judgment_area_x1_max,judgment_area_y1_max),(judgment_area_x1_min,judgment_area_y1_min),(255, 255, 255), flag_point_list[0])
    cv2.rectangle(imshow,  (judgment_area_x2_max,judgment_area_y2_max),(judgment_area_x2_min,judgment_area_y2_min),(255, 255, 255), flag_point_list[1])
    cv2.rectangle(imshow,  (judgment_area_x3_max,judgment_area_y3_max),(judgment_area_x3_min,judgment_area_y3_min),(255, 255, 255), flag_point_list[2])
    cv2.imshow("test",imshow)
    k = cv2.waitKey(5)&0xff
    if k==27:
        break
