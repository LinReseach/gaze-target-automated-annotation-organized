#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:18:26 2024

@author: chenglinlin
"""
import cv2
import os


def frames_to_video(input_folder, output_video, fps):
    # Get all the image filenames in the input folder
    image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Sort the image files based on their names
    image_files.sort()

    # Read the first image to get the width and height
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' for .avi format
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each frame to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)

    # Release the video writer object
    out.release()

if __name__ == "__main__":
    # Path to the folder containing the frames
    input_folder = '/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/40_2_front_l2cs/'

    # Path to the output video file
    output_video = "/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/40_2_front_l2cs.mp4"

    # Frames per second (FPS) of the output video
    fps = 24

    # Call the function to convert frames to video
    frames_to_video(input_folder, output_video, fps)