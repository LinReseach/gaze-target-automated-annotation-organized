#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:45:43 2024

@author: chenglinlin
"""

import cv2
filename='/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/videos/40_2_front.avi'
picturename='/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/frames/40_2_front/'
# vidcap = cv2.VideoCapture(filename)
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   cv2.imwrite(picturename+"frame%d.jpg" % count, image)     # save frame as JPEG file
#   if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#       break
#   count += 1
  
  
import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total Frames:", total_frames)

    # Loop through each frame
    current_frame = 0
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        frame_name = f"frame_{current_frame}.jpg"
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, frame)

        # Print progress
        print(f"Extracting frame {current_frame}/{total_frames}")

        # Move to the next frame
        current_frame += 1

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the input video file
    video_path = filename

    # Output folder where frames will be saved
    output_folder = picturename

    # Call the function to extract frames
    extract_frames(video_path, output_folder)
