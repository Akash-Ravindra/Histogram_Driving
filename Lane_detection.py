# %%
import cv2 as cv
import numpy as np
show = True
save_video = False
# %%
def showimage(image,label = "image"):
    cv.imshow(label,image)
    cv.waitKey()
    cv.destroyAllWindows()

# %%
def length_of_line(line):
    x1,y1,x2,y2 = line[0]
    return np.sqrt((x2-x1)**2+(y2-y1)**2)
def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    left_length = []
    right_length = []
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x1 == x2:
                    continue
                slope = (y2-y1)/(x2-x1)
                intercept = y1-slope*x1
                length = length_of_line([[x1,y1,x2,y2]])
                if slope < 0:
                    left_lines.append((slope,intercept))
                    left_length.append(length)
                    left_weights.append((1.0/length))
                else:
                    right_lines.append((slope,intercept))
                    right_length.append(length)
                    right_weights.append((1.0/length))
    except:
        left_lines.clear()
        right_lines.clear()
    try:
        left_lane = np.dot(left_weights,left_lines) / np.sum(left_weights)
        left_avg_length = np.mean(np.asarray(left_length),axis = 0)
    except:
        pass
    try:
        right_lane = np.dot(right_weights,right_lines) / np.sum(right_weights)
        right_avg_length = np.mean(np.asarray(right_length),axis=0)
    except:
        pass
    return left_lane,right_lane,left_avg_length,right_avg_length

# %%
video_handler = cv.VideoCapture('./Data/whiteline.mp4')
if (video_handler.isOpened() == False):
    print("Error opening the video file")
else:
# Get frame rate information
    fps = int(video_handler.get(5))
    print("Frame Rate : ",fps,"frames per second")	
    # Get frame count
    frame_count = video_handler.get(7)
    print("Frame count : ", frame_count)

frame_width = int(video_handler.get(3))
frame_height = int(video_handler.get(4))
   


if(save_video):
    size = (frame_width, frame_height)
    result = cv.VideoWriter('Question_2_result.avi', 
                            cv.VideoWriter_fourcc(*'MJPG'),
                            25, size)

# %%
frames = []
i = 0
previous_tag=None
previous_lanes = []
while(video_handler.isOpened()):
    # nonoise_vid.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    ret, frame = video_handler.read()
    if ret == True:
        frames.append(frame)
        
        ## Convert Blur image and apply adaptive histogram equalization
        blurred = cv.GaussianBlur(frame,(5,5),0)
        l,a,b = cv.split(cv.cvtColor(blurred,cv.COLOR_BGR2LAB))
        clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        adpt_hist = clahe.apply(l)
        corrected_image = cv.cvtColor((cv.merge((adpt_hist,a,b))),cv.COLOR_LAB2BGR)
        
        ## conver image to gray scale
        gray = cv.cvtColor(corrected_image,cv.COLOR_BGR2GRAY)
        ## apply thresholding
        ret, thresh = cv.threshold(gray,150,256,cv.THRESH_BINARY)
        ## Apply canny edge dector
        edges_image = cv.Canny(thresh,50,150,apertureSize = 3)
        # showimage(thresh)
        
        ## Define the polygon that contains the area of interest
        ROI_points = np.array([[(150,frame.shape[0]),(450,310),(490,310),(880,frame.shape[0])]])
        masked_image = np.zeros_like(edges_image)
        mask = cv.fillPoly(masked_image,ROI_points,255)
        ## Apply mask to edges image
        masked_image = cv.bitwise_and(edges_image,mask)
        # showimage(masked_image)
        
        ## Apply Hough transform to find the lines
        lines = cv.HoughLinesP(masked_image,1,np.pi/180,100,np.array([]),minLineLength=70,maxLineGap=15)
        frame_cpy = np.copy(frame)
        # try:
        #     for line in lines:
        #         [x1,y1,x2,y2] = line[0]
        #         cv.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
        # except:
        #     pass
        right_line = []
        left_line = []
        draw_lines = []
        
        
        left_lane,right_lane,left_avg_length,right_avg_length = average_slope_intercept(lines)
        
        if(np.isnan(left_lane).any() or left_lane.size == 0):
            left_lane = previous_lanes[0][0]
            left_avg_length = previous_lanes[0][2]
        if(np.isnan(right_lane).any() or right_lane.size == 0):
            right_lane = previous_lanes[0][1]
            right_avg_length = previous_lanes[0][3]
            
            
        previous_lanes.clear()
        previous_lanes.append([left_lane,right_lane,left_avg_length,right_avg_length])
    
        lines_of_interest = []    
        ## Find the lines that are solid or dashed
        if(right_avg_length > left_avg_length):
            solid_lines = right_lane
            dashed_lines = left_lane
        else:
            solid_lines = left_lane
            dashed_lines = right_lane
        lines_of_interest.append(solid_lines)
        lines_of_interest.append(dashed_lines)
        # print("leftLength: ",left_avg_length,"rightLength: ",right_avg_length)
        ## Find the endpoints of the line, and draw the line
        for line in lines_of_interest:
            slope = line[0]
            intercept = line[1]
            start_point = (frame.shape[0]-intercept)/slope
            start_point = (int(start_point),frame.shape[0])
            endpoint = (350-intercept)/slope
            endpoint = (int(endpoint),350)
            draw_lines.append([start_point,endpoint])
        frame_cpy = np.copy(frame)
        cv.line(frame_cpy, draw_lines[0][0], draw_lines[0][1], (0,255,0), 3, cv.LINE_AA)
        cv.line(frame_cpy, draw_lines[1][0], draw_lines[1][1], (0,0,255), 3, cv.LINE_AA)
        
        ################################################
        if show:
            cv.namedWindow('frame')
            cv.imshow('Frame', frame_cpy)
            if cv.waitKey(0) & 0xFF == ord('s'):
                cv.destroyAllWindows()
                break
        #############################################
        # # cv.namedWindow('thresh')
        # # cv.imshow('thresh', masked_image)
        
        # ## Write the frame to the output video
        if(save_video):
            result.write(frame_cpy)
        i += 1
    else:
        cv.waitKey(1)
        cv.destroyAllWindows()
        break


