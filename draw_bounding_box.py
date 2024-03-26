#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:46:58 2024

@author: chenglinlin
"""

import cv2

# Function to draw bounding boxes on video frames
def draw_bounding_boxes(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    # Define the bounding boxes for two objects (x, y, width, height)
#    bounding_box1 = (114, 149, 96,290)# for video 40_2_front
#    bounding_box2 = (233, 337, 162,100)
    bounding_box1 = (123, 135, 134,325)
    bounding_box2 = (313,410, 185,67)
    # Colors for the bounding boxes (BGR format)
    color2 = (255, 0, 0)  # Blue
    color1 = (0, 255, 0)  # Green
    
    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw the first bounding box
        x1, y1, w1, h1 = bounding_box1
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), color1, 2)
        
        # Draw the second bounding box
        x2, y2, w2, h2 = bounding_box2
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), color2, 2)
        
        # Write the frame with bounding boxes to the output video
        out.write(frame)
        
        # Display the frame
        cv2.imshow('Video with Bounding Boxes', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the input video file
    video_path ='/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/videos/32_2_front.avi'
    
    # Path to save the output video file
    output_path = '/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/videos_bb/32_2_front.avi'
    
    # Call the function to draw bounding boxes on the video
    draw_bounding_boxes(video_path, output_path)
