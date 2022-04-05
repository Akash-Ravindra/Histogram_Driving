# %%
import cv2 as cv
import numpy as np
show = True
save_video = False
# %%
video_handler = cv.VideoCapture('./Data/challenge.mp4')
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

output_frame = np.zeros((frame_height+100,frame_width+700,3),dtype=np.uint8)



if(save_video):
    size = (output_frame.shape[1], output_frame.shape[0])
    result = cv.VideoWriter('Question_3_result.avi', 
                            cv.VideoWriter_fourcc(*'MJPG'),
                            25, size)

# %%
frames = []
frame_num = 0
per_frame_curvature = []
previous_tag=None
while(video_handler.isOpened()):
    # nonoise_vid.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    ret, frame = video_handler.read()
    if ret == True:
        frames.append(frame)
        blurred = cv.GaussianBlur(frame,(5,5),0)
        l,a,b = cv.split(cv.cvtColor(blurred,cv.COLOR_BGR2LAB))
        clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        adpt_hist = clahe.apply(l)
        corrected_image = cv.cvtColor((cv.merge((adpt_hist,a,b))),cv.COLOR_LAB2BGR)
        # showimage(corrected_image)
        
        
        reshape_point = np.float32([[0,700],[0,0],[500,0],[500,700]])
        ROI_points = np.float32([[240,660],[590,450],[730,450],[1120,660]])
        warping_matrix = cv.getPerspectiveTransform(ROI_points,reshape_point)
        warped_image = cv.warpPerspective(corrected_image,warping_matrix,(500,700))
        # showimage(warped_image)
        
        _,thresh = cv.threshold(warped_image[:,:,-1],240,255,cv.THRESH_BINARY)
        combined_thresh = thresh
        # showimage(thresh)
        ## Find the columnar histogram of the image for the sliding windows
        hist = np.sum(combined_thresh,axis=0)/255
        left_current_x = np.argmax(hist[:hist.shape[0]//2])
        right_current_x = np.argmax(hist[hist.shape[0]//2:])+hist.shape[0]//2
        
        ## Find the left and right lane lines using the sliding window technique
        number_of_windows = 30
        window_height = combined_thresh.shape[0]//number_of_windows
        white_pixels = np.nonzero(combined_thresh)
        white_pixels_y = np.array(white_pixels[0])
        white_pixels_x = np.array(white_pixels[1])
        left_lane_idx = []
        right_lane_idx = []
        for i in range(number_of_windows):
            ## Definition of the window dimensions
            lim_y_top = frame.shape[0]-(i+1)*window_height
            lim_y_bottom = frame.shape[0]-i*window_height
            ## Definition of the window x-coordinates left and right
            lim_x_left_low = left_current_x-(window_height//2)
            lim_x_left_high = left_current_x+(window_height//2)
            lim_x_right_low = right_current_x-(window_height//2)
            lim_x_right_high = right_current_x+(window_height//2)
            
            ## Finding all the white pixels in the window
            pixels_in_left_window = np.nonzero((white_pixels_y>=lim_y_top) & (white_pixels_y<lim_y_bottom)
                                               & (white_pixels_x>=lim_x_left_low) & (white_pixels_x<lim_x_left_high))[0]
            pixels_in_right_window = np.nonzero((white_pixels_y>=lim_y_top) & (white_pixels_y<lim_y_bottom)
                                               & (white_pixels_x>=lim_x_right_low) & (white_pixels_x<lim_x_right_high))[0]
            
            left_lane_idx.append(pixels_in_left_window)
            right_lane_idx.append(pixels_in_right_window)
            
            ## Updating the current x-coordinates
            if(len(pixels_in_left_window)>500):
                left_current_x = np.mean(white_pixels_x[pixels_in_left_window])
                left_previous_x = left_current_x
            ## Updating the current x-coordinates
            if(len(pixels_in_right_window)>20):
                right_current_x = np.mean(white_pixels_x[pixels_in_right_window])
                right_previous_x = right_current_x
            pass   
        
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)
        ## Finding the pixels in the left and right lanes
        left_lane_x = white_pixels_x[left_lane_idx]
        left_lane_y = white_pixels_y[left_lane_idx]
        right_lane_x = white_pixels_x[right_lane_idx]
        right_lane_y = white_pixels_y[right_lane_idx]
        warped_image_cpy_2 = warped_image.copy()
        
        
        
        left_xy = np.concatenate((left_lane_x.reshape(left_lane_x.shape[0],1),left_lane_y.reshape(left_lane_y.shape[0],1)),axis=1)
        right_xy = np.concatenate((right_lane_x.reshape(right_lane_x.shape[0],1),right_lane_y.reshape(right_lane_y.shape[0],1)),axis=1)
        ## Plotting every single white pixel believed to be lane lines found in the warped image
        plot_img = cv.polylines(warped_image_cpy_2,left_xy.reshape(-1,1,2),True,(0,0,255),5)
        plot_img = cv.polylines(warped_image_cpy_2,right_xy.reshape(-1,1,2),True,(0,0,255),5)
        
        ## Fitting a polynomial to the left and right lanes
        ## Left lane
        sub_left = np.polynomial.polynomial.Polynomial.fit(left_lane_y,left_lane_x,2)
        x_l,y_l = sub_left.linspace(100,domain=[100,680])
        all_points = np.concatenate((y_l.reshape(y_l.shape[0],1),x_l.reshape(x_l.shape[0],1)),axis=1)
        ## Finding the curvature of the left lane
        real_points = cv.perspectiveTransform(all_points.reshape(-1,1,2).astype(np.float32),np.linalg.inv(warping_matrix)).reshape(-1,2).astype(np.int32)
        real_curve = np.polynomial.polynomial.Polynomial.fit(real_points[:,0]*(10/frame.shape[0]),real_points[:,1]*(3.7/frame.shape[1]),2)
        curvature_L = np.float_power((1+np.power(real_curve.deriv(1)(300),2)),(3/2))/(np.abs(real_curve.deriv(2)(0)))
        
        ## Right lane
        sub_right = np.polynomial.polynomial.Polynomial.fit(right_lane_y,right_lane_x,2)
        x_r,y_r = sub_right.linspace(100,domain=[100,680])
        x_r = x_r[::-1]
        y_r = y_r[::-1]
        all_points = np.concatenate((all_points,np.concatenate((y_r.reshape(y_r.shape[0],1),x_r.reshape(x_r.shape[0],1)),axis=1)),axis=0).astype(np.int32)
        
        
        ## Finding the curvature of the right lane
        right_points = np.concatenate((y_r.reshape(y_r.shape[0],1),x_r.reshape(x_r.shape[0],1)),axis=1)
        real_points = cv.perspectiveTransform(right_points.reshape(-1,1,2).astype(np.float32),np.linalg.inv(warping_matrix)).reshape(-1,2).astype(np.int32)
        real_curve = np.polynomial.polynomial.Polynomial.fit(real_points[:,0]*(10/frame.shape[0]),real_points[:,1]*(3.7/frame.shape[1]),2)
        curvature_R = np.float_power((1+np.power(real_curve.deriv(1)(300),2)),(3/2))/(np.abs(real_curve.deriv(2)(0)))
        # print('curvature_L',curvature_L,"   curvature_R",curvature_R, "  Average",((3*curvature_L)+curvature_R)/4,i)
        per_frame_curvature.append([curvature_L,curvature_R])
        warped_image_cpy = warped_image.copy()
        plot_img = cv.polylines(warped_image_cpy,all_points.reshape(-1,1,2),True,(255,0,0),10)
        
        final_points = cv.perspectiveTransform(all_points.reshape(-1,1,2).astype(np.float32),np.linalg.inv(warping_matrix)).reshape(-1,2).astype(np.int32)
        
        frame_cpy = frame.copy()
        ## Draw the polygon onto the lane
        cv.fillPoly(frame_cpy,pts = [final_points],color = (255,255,0))
        cv.addWeighted(frame_cpy,0.5,frame,0.5,0,frame)
        ## Finding the average curvature of the left and right lanes every 5 frames
        if(frame_num%5==0):
            average = np.mean(np.mean(np.asarray(per_frame_curvature),axis=0),axis=0)
            per_frame_curvature.clear()
            print('Average Curvature is',average)
            
        frame_num+=1
        
        # output_frame = np.zeros((frame.shape[0]+100,frame.shape[1]+700,3),dtype=np.uint8)
        output_frame[0:frame.shape[0],0:frame.shape[1]] = frame
        ## Resizing the warped image to the output frame
        warped_image = cv.resize(warped_image,(700//2,720//2))
        output_frame[0:360,frame.shape[1]:frame.shape[1]+350] = warped_image
        ## Resizing the plot image to the output frame
        combined_thresh = cv.resize(combined_thresh,(700//2,720//2))
        combined_thresh = np.dstack((combined_thresh,combined_thresh,combined_thresh))
        output_frame[0:360,frame.shape[1]+350:] = combined_thresh
        ## Resizing the warped plotted image to the output frame
        warped_image_cpy = cv.resize(warped_image_cpy,(700//2,720//2))
        output_frame[360:720,frame.shape[1]:frame.shape[1]+350] = warped_image_cpy
        ## Resizing the plot threshold image to the output frame
        warped_image_cpy_2 = cv.resize(warped_image_cpy_2,(700//2,720//2))
        output_frame[360:720,frame.shape[1]+350:] = warped_image_cpy_2
        ## Text space
        bottom_image = np.ones((100,frame.shape[1]+700,3),dtype=np.uint8)
        bottom_image[:,:,1] = 127
        
        
        cv.putText(bottom_image,"Average Curvature = "+str(average),(10,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        if(average<2500):
            direction = "Right Turn"
        else:
            direction = "Straight"
        cv.putText(bottom_image,"Direction = "+direction,(800,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv.putText(bottom_image,"LLane Curvature = "+str(curvature_L),(10,75),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv.putText(bottom_image,"RLane Curvature = "+str(curvature_R),(800,75),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        output_frame[720:,:] = bottom_image
        
        if save_video:
            result.write(output_frame)
        if show:
            cv.imshow('Output Frame', output_frame)
            if cv.waitKey(0) & 0xFF == ord('s'):
                cv.destroyAllWindows()
                break
        
    else:
        cv.waitKey(1)
        cv.destroyAllWindows()         
        break


# %%



