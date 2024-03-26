import argparse, os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
from scipy.misc import imresize
from skimage.transform import resize

from face_detection import RetinaFace
from model import ModelSpatial
from utils import imutils, evaluation
from config import *
from readbbtxt import readbbtxt as readvis
from readbbtxt_inv import readbbtxt as readinvis

# from face_detection import RetinaFace

# /home/linlincheng/Documents/Projects/attention-target-detection-master/data/demo/Proefpersoon_63001_video
# [Proefpersoon11044_sessie1_uitleg, Proefpersoon_63001_video, Proefpersoon_63009_video_1, Proefpersoon_63009_video_2]

NOFACE = 42
datafolder = './data_fin/'
vis_datafile = 'pixel_position_vis.txt'
invis_datafile = 'pixel_position_invis_new.txt'
vis_data = readvis(datafolder + vis_datafile)
invis_data = readinvis(datafolder + invis_datafile)
# remove .png extension from filenames
vis_data['file'] = vis_data['file'].apply(lambda x: x[:-4])
invis_data['file'] = invis_data['file'].apply(lambda x: x[:-4])



def parse_args():
    """Parse input arguments."""
    """model parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
    # parser.add_argument('--image_dir', type=str, help='images', default='data/all/Proefpersoon11044_sessie1_uitleg')
    # parser.add_argument('--head', type=str, help='head bounding boxes',
    #                     default='data/all/Proefpersoon11044_sessie1_uitleg.txt')
    parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')
    parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=150)
    args = parser.parse_args()
    return args


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def inbb(tlx, tly, brx, bry, gazex, gazey):
    return tlx <= gazex <= tlx + (brx - tlx) and tly <= gazey <= tly + (bry - tly)


def get_classname(id):
    if id == 0:
        return 'Pen and paper'
    elif id == 1:
        return 'robot'
    elif id == 2:
        return 'tablet'
    elif id == 3:
        return 'elsewhere'
    elif id == 4:
        return 'unknown'


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Returns the coordinates px, py. p is the intersection of the lines
    defined by ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
    """
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return int(px), int(py)


def line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    This is a general function to calculate the intersection point between four points (two line segments)

    Returns the coordinate intersection_x, intersection_y.
    defined by ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
    if no intersection, return None
    """

    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
    num1 = (x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3)
    num2 = (x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)

    if denom == 0:  # two line segments are parallel or coincident
        return None

    t1 = num1 / denom
    t2 = num2 / denom

    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        intersection_x = round(x1 + t1 * (x2 - x1))
        intersection_y = round(y1 + t1 * (y2 - y1))
        return intersection_x, intersection_y
    else:
        return None


def intersection_proportion(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    This is a function to calculate the proportion of the intersected segments in the bounding box area

    Return a number that represent the percentage of the proportion
    """
    box_width = x2 - x1
    box_height = y2 - y1
    diagonal_length = (box_height ** 2 + box_width ** 2) ** 0.5

    delta_x = max(x3, x4) - min(x3, x4)
    delta_y = max(y3, y4) - min(y3, y4)
    intersection_length = (delta_y ** 2 + delta_y ** 2) ** 0.5

    proportion = intersection_length / diagonal_length * 100

    return proportion

def get_extended_point(height, width, x11, y11, x22, y22):


    if x22 - x11 != 0:

        slope_lin = (y22 - y11) / (x22 - x11)
        

    else:
        slope_lin=0
        x_bottom = x11
        y_bottom = y11
 
	
    # Calculate slope of the line


    # Calculate y-intercept
    y_intercept = y11 - slope_lin * x11

    # Calculate new points to extend the line to image boundaries
    if slope_lin != 0:
        # Extend to top boundary (y = 0)
        x_top = int(-y_intercept / slope_lin)
        y_top = 0

        # Extend to bottom boundary (y = height - 1)
        x_bottom = int((height - 1 - y_intercept) / slope_lin)
        y_bottom = height - 1
    else:
        # For vertical lines, x coordinates remain the same
        x_top = x11
        x_bottom = x11

        # Extend to top and bottom boundaries
        y_top = 0
        y_bottom = height - 1
    

    return x_bottom,y_bottom

def classify_line_box_intersection_proportion(face_x,face_y,image_in, gazex, gazey, face, vid_type, filename):
    """
    - This is the function to classify the intersection points between gaze and bounding boxs
    *********  with considering the proportion of intersection!!! ***********
    - The sequence of consideration is tablet, pen & paper, and then the robot. (based on their size in image. this
    might be able to increase the accuracy of the classification in the case of objects overlapping)

    Return one of five numbers (0-4) which represents the object label.
    """
    # print(face)
    intersection_points_tablet = []
    intersection_points_pp = []
    intersection_points_robot = []

    if face == 0 or None:
        class_pred = 4
        #return 4

    (h, w) = image_in.shape[:2]
    #print(h,w)
    pos = (face_x, face_y)
    # print(face_x, face_y)
    gazex_norm = round(gazex)
    gazey_norm = round(gazey)
    # print(gazex,gazex_norm)

    # print(f'gazex, gazey = {(gazex, gazey)}')

    try:
        gaze_slope = float((gazey_norm - pos[1]) / (gazex_norm - pos[0]))
        # print(gaze_slope)
        # print(f"gaze slope: {gaze_slope}")
    except ZeroDivisionError:
        gaze_slope = 1.0
    gaze_intercept = float(pos[1] - gaze_slope * pos[0])
    # print(f"gaze intercept: {gaze_intercept}")
    gazex_ext = round(pos[0] + gaze_slope*100)
    gazey_ext = round(pos[1] + gaze_slope*100)
    gazex_ext = round(pos[0] + 3*(gazex_norm-pos[0]))
    gazey_ext = round(pos[1] + 3*(gazey_norm-pos[1]))
    image_edges = [(0, 0), (w, 0), (w, h), (0, h)]
    x1, y1 = round(pos[0]), round(pos[1])
    # x2_norm, y2_norm = gazex_norm, gazey_norm
    #x2_ext, y2_ext = gazex_ext, gazey_ext
    x2_ext, y2_ext=get_extended_point(h, w, pos[0], pos[1], gazex_norm, gazey_norm)
    #print(w,h,x2_ext,y2_ext)
    # print(x2_ext,y2_ext)

    if vid_type == 'vis/':
        rec = vis_data[vis_data['file'] == filename].iloc[0]
    elif vid_type == 'invis/':
        rec = invis_data[invis_data['file'] == filename].iloc[0]
    else:
        rec = None
        print('Video type error!')

    x_left_tab = int(rec['tl_tablet_x'] * w)
    x_right_tab = int(rec['br_tablet_x'] * w)
    y_bottom_tab = int(rec['br_tablet_y'] * h)
    y_top_tab = int(rec['tl_tablet_y'] * h)

    x_left_pp = int(rec['tl_pp_x'] * w)
    x_right_pp = int(rec['br_pp_x'] * w)
    y_bottom_pp = int(rec['br_pp_y'] * h)
    y_top_pp = int(rec['tl_pp_y'] * h)

    x_left_rbt = int(rec['tl_robot_x'] * w)
    x_right_rbt = int(rec['br_robot_x'] * w)
    y_bottom_rbt = int(rec['br_robot_y'] * h)
    y_top_rbt = int(rec['tl_robot_y'] * h)

    box_corners_tablet = [(x_left_tab, y_bottom_tab), (x_right_tab, y_bottom_tab), (x_right_tab, y_top_tab),
                          (x_left_tab, y_top_tab)]
    box_corners_pp = [(x_left_pp, y_bottom_pp), (x_right_pp, y_bottom_pp), (x_right_pp, y_top_pp),
                      (x_left_pp, y_top_pp)]
    box_corners_robot = [(x_left_rbt, y_bottom_rbt), (x_right_rbt, y_bottom_rbt), (x_right_rbt, y_top_rbt),
                         (x_left_rbt, y_top_rbt)]



    # to check if the original gaze end point is in the bbox or not, if not then check intersections for extended gaze
    if inbb(x_left_tab, y_top_tab, x_right_tab, y_bottom_tab, gazex_norm, gazey_norm):
        class_pred=2
        #return 2
    elif inbb(x_left_pp, y_top_pp, x_right_pp, y_bottom_pp, gazex_norm, gazey_norm):
        class_pred = 0
        #return 0
    elif inbb(x_left_rbt, y_top_rbt, x_right_rbt, y_bottom_rbt, gazex_norm, gazey_norm):
        class_pred = 1
        #return 1
    else:

        # check if there is any the intersection point in tablet box
        for i in range(4):
            x1_t, y1_t = box_corners_tablet[i]
            x2_t, y2_t = box_corners_tablet[(i + 1) % 4]  # Next corner (wraps around to the first corner)
            # print((x1_t, y1_t))
            # print((x2_t, y2_t))
            # Check if the line and line segment are parallel
            try:
                # tablet_edge_slope = round((y2_t - y1_t) / (x2_t - x1_t))
                tablet_edge_slope = abs((y2_t - y1_t) / (x2_t - x1_t))
                # print(f"tablet edge slope: {tablet_edge_slope}")
            except ZeroDivisionError:
                tablet_edge_slope = 1.0
                # print(f"tablet edge slope: {tablet_edge_slope}")

            if gaze_slope == tablet_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t, y2_t) is not None:
                intersection_x_t, intersection_y_t = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_t, y1_t, x2_t,
                                                                               y2_t)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_t, x2_t) <= abs(intersection_x_t) <= max(x1_t, x2_t) and min(y1_t, y2_t) <= abs(
                        intersection_y_t) <= max(y1_t, y2_t):
                    intersection_points_tablet.append((intersection_x_t, intersection_y_t))
                    cv2.circle(image_in, (intersection_x_t, intersection_y_t), 10, (0, 0, 255), 1)

            # if intersection_points_tablet:
            #     print(f"tablet: {intersection_points_tablet}")
        # return intersection_points_tablet

        # check if there is any the intersection point in pp box
        for i in range(4):
            x1_p, y1_p = box_corners_pp[i]
            x2_p, y2_p = box_corners_pp[(i + 1) % 4]  # Next corner (wraps around to the first corner)

            # Check if the line and line segment are parallel
            try:
                # pp_edge_slope = round((y2_p - y1_p) / (x2_p - x1_p))
                pp_edge_slope = (y2_p - y1_p) / (x2_p - x1_p)
            except ZeroDivisionError:
                pp_edge_slope = 1.0

            if gaze_slope == pp_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p, y2_p) is not None:
                intersection_x_p, intersection_y_p = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_p, y1_p, x2_p,
                                                                               y2_p)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_p, x2_p) <= abs(intersection_x_p) <= max(x1_p, x2_p) and min(y1_p, y2_p) <= abs(
                        intersection_y_p) <= max(y1_p, y2_p):
                    intersection_points_pp.append((intersection_x_p, intersection_y_p))
                    cv2.circle(image_in, (intersection_x_p, intersection_y_p), 10, (255, 0, 0), 1)


        # check if there is any the intersection point in robot box
        for i in range(4):
            x1_r, y1_r = box_corners_robot[i]
            x2_r, y2_r = box_corners_robot[(i + 1) % 4]  # Next corner (wraps around to the first corner)

            # Check if the line and line segment are parallel
            try:
                # robot_edge_slope = round((y2_r - y1_r) / (x2_r - x1_r))
                robot_edge_slope = (y2_r - y1_r) / (x2_r - x1_r)
            except ZeroDivisionError:
                robot_edge_slope = 1.0

            if gaze_slope == robot_edge_slope:
                continue

            if line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r, y2_r) is not None:
                intersection_x_r, intersection_y_r = line_segment_intersection(x1, y1, x2_ext, y2_ext, x1_r, y1_r, x2_r,
                                                                               y2_r)

                # Check if the intersection point lies within the range of the line segment
                if min(x1_r, x2_r) <= abs(intersection_x_r) <= max(x1_r, x2_r) and min(y1_r, y2_r) <= abs(
                        intersection_y_r) <= max(y1_r, y2_r):
                    intersection_points_robot.append((intersection_x_r, intersection_y_r))
                    cv2.circle(image_in, (intersection_x_r, intersection_y_r), 10, (0, 255, 0), 1)


        class_pred = 3
        prop_tab = 0
        prop_pp = 0
        prop_rbt = 0
        max_prop = 0
        if len(intersection_points_tablet) == 2:
            class_pred = 2
            prop_tab = intersection_proportion(x_left_tab, y_top_tab, x_right_tab, y_bottom_tab,
                                               intersection_points_tablet[0][0], intersection_points_tablet[0][1],
                                               intersection_points_tablet[1][0], intersection_points_tablet[1][1])
        elif len(intersection_points_tablet) == 1:
            class_pred = 2
        else:
            pass

        if len(intersection_points_pp) == 2:
            class_pred = 0
            prop_pp = intersection_proportion(x_left_pp, y_top_pp, x_right_pp, y_bottom_pp,
                                              intersection_points_pp[0][0], intersection_points_pp[0][1],
                                              intersection_points_pp[1][0], intersection_points_pp[1][1])
        elif len(intersection_points_pp) == 1:
            class_pred = 0
        else:
            pass

        if len(intersection_points_robot) == 2:
            class_pred = 1
            prop_rbt = intersection_proportion(x_left_rbt, y_top_rbt, x_right_rbt, y_bottom_rbt,
                                               intersection_points_robot[0][0], intersection_points_robot[0][1],
                                               intersection_points_robot[1][0], intersection_points_robot[1][1])
        elif len(intersection_points_tablet) == 1:
            class_pred = 1
        else:
            pass

        # to find the max proportion among the three intersections
        if prop_tab > max_prop:
            max_prop = prop_tab
            class_pred = 2
        if prop_pp > max_prop:
            max_prop = prop_pp
            class_pred = 0
        if prop_rbt > max_prop:
            max_prop = prop_rbt
            class_pred = 1

        # print(max_prop)
        #return class_pred,x2_ext,y2_ext
    return class_pred, x2_ext, y2_ext

#gaze_class = classify_gaze(result_folder, height, width, gazex, gazey,facex,facey,frame_raw,face_status,folder)
def classify_gaze2(vid_type, h, w, gazex, gazey,face_x,face_y,image_in,face,filename):
    """
    returns
    0 : pen & paper
    1 : robot
    2 : tablet
    3 : elsewhere
    4 : unknown
    """
    # class_type, x2_ext, y2_ext = classify_line_box_intersection_proportion(face_x, face_y, image_in, gazex, gazey, face,
    #                                                                        vid_type, filename)
    # return class_type, 1, x2_ext, y2_ext
    if gazex == NOFACE and gazey == NOFACE:
        return 4,0

    if vid_type == 'vis/':
        rec = vis_data[vis_data['file'] == folder].iloc[0]
    elif vid_type == 'invis/' or vid_type == 'object/':
        rec = invis_data[invis_data['file'] == folder].iloc[0]
    else:
        print("Encountering a video type error.")

    # Check small objects first because if e.g. tablet bb intersects with bb of robot and gaze is towards the parts of
    # intersection, changes are higher the child is indeed looking at the smaller bb, thus the tablet.

    if inbb(int(rec['tl_tablet_x'] * w), int(rec['tl_tablet_y'] * h),
            int(rec['br_tablet_x'] * w), int(rec['br_tablet_y'] * h), gazex, gazey):
        return 2,0,0,0
    elif inbb(int(rec['tl_pp_x'] * w), int(rec['tl_pp_y'] * h), int(rec['br_pp_x'] * w),
              int(rec['br_pp_y'] * h), gazex, gazey):
        return 0,0,0,0
    elif inbb(int(rec['tl_robot_x'] * w), int(rec['tl_robot_y'] * h),
              int(rec['br_robot_x'] * w), int(rec['br_robot_y'] * h), gazex, gazey):
        return 1,0,0,0
    else:
        #print('extend')
        class_type,x2_ext,y2_ext = classify_line_box_intersection_proportion(face_x,face_y,image_in, gazex,gazey, face, vid_type, filename)
        return class_type,1,x2_ext,y2_ext

def classify_gaze(vid_type, h, w, gazex, gazey):
    """
    returns
    0 : pen & paper
    1 : robot
    2 : tablet
    3 : elsewhere
    4 : unknown
    """

    if gazex == NOFACE and gazey == NOFACE:
        return 4

    if vid_type == 'vis/':
        rec = vis_data[vis_data['file'] == folder].iloc[0]
    elif vid_type == 'invis/' or vid_type == 'object/':
        rec = invis_data[invis_data['file'] == folder].iloc[0]
    else:
        print("Encountering a video type error.")

    # Check small objects first because if e.g. tablet bb intersects with bb of robot and gaze is towards the parts of
    # intersection, changes are higher the child is indeed looking at the smaller bb, thus the tablet.

    if inbb(int(rec['tl_tablet_x'] * w), int(rec['tl_tablet_y'] * h),
            int(rec['br_tablet_x'] * w), int(rec['br_tablet_y'] * h), gazex, gazey):
        return 2
    elif inbb(int(rec['tl_pp_x'] * w), int(rec['tl_pp_y'] * h), int(rec['br_pp_x'] * w),
              int(rec['br_pp_y'] * h), gazex, gazey):
        return 0
    elif inbb(int(rec['tl_robot_x'] * w), int(rec['tl_robot_y'] * h),
              int(rec['br_robot_x'] * w), int(rec['br_robot_y'] * h), gazex, gazey):
        return 1
    else:
        return 3
def get_face_info(frame_name, face_frame):
    detector = RetinaFace(gpu_id=0)
    faces = detector(face_frame)

    biggest_width = 0
    biggest_height = 0
    x_min_ = 0
    x_max_ = 0
    y_min_ = 0
    y_max_ = 0
    face_status = False

    for box, landmarks, score in faces:
        if score < .85:  # default .95
            continue
        x_min = int(box[0])
        if x_min < 0:
            x_min = 0
        y_min = int(box[1])
        if y_min < 0:
            y_min = 0
        x_max = int(box[2])
        y_max = int(box[3])
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        if bbox_width > biggest_width:
            biggest_width = bbox_width
            biggest_height = bbox_height
            x_min_ = x_min
            x_max_ = x_max
            y_min_ = y_min
            y_max_ = y_max
            face_status = True

    face_box = [x_min_, y_min_, x_max_, y_max_]
    # print(face_box)
    if result_folder == 'vis/':
        facepath = './data/face_vis/'
    else:
        facepath = './data/face_invis/'

    with open(os.path.join(facepath, f'{folder}.txt'), 'a') as file:
        file.write(frame_name)
        file.write(',')
        file.write(','.join(str(round(float(b))) for b in face_box))
        file.write(',')
        file.write(str(face_status))
        file.write('\n')



def run():
    print(f'{folder} is in processing...')
    if not os.path.exists(head_info):
        img_list = [img for img in os.listdir(image_dir)]
        img_list = sorted(img_list)
        #print(img_list[0:5])
        for img in img_list:
            im = cv2.imread(os.path.join(image_dir, img))
            # im = Image.open(os.path.join(args.image_dir, img))
            # im = im.convert('RGB')
            get_face_info(img, im)
        print(f"face detection for {folder} is done!")

    column_names = ['frame', 'left', 'top', 'right', 'bottom', 'face']
    df = pd.read_csv(head_info, names=column_names, index_col=0)

    # set up data transformation
    test_transforms = _get_transform()

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    pred_x_list = []
    pred_y_list = []
    gaze_class_list = []
    face_list = []

    with torch.no_grad():
        for i in df.index:
            start_fps = time.time()
            _,image_in= cv2.VideoCapture(os.path.join(image_dir, i)).read()
            print(os.path.join(image_dir, i),type(image_in))
            frame_raw = Image.open(os.path.join(image_dir, i))
            frame_raw = frame_raw.convert('RGB')

            width, height = frame_raw.size

            if df.loc[i, 'face']:
                face_status = 1
                head_box = [round(float(df.loc[i, 'left'])), round(float(df.loc[i, 'top'])),
                            round(float(df.loc[i, 'right'])), round(float(df.loc[i, 'bottom']))]

                head = frame_raw.crop((head_box))  # head crop

                head = test_transforms(head)  # transform inputs
                frame = test_transforms(frame_raw)
                head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width,
                                                            height, resolution=input_resolution).unsqueeze(0)

                head = head.unsqueeze(0).cuda()
                frame = frame.unsqueeze(0).cuda()
                head_channel = head_channel.unsqueeze(0).cuda()

                # forward pass
                raw_hm, _, inout = model(frame, head_channel, head)

                # heatmap modulation
                raw_hm = raw_hm.cpu().detach().numpy() * 255
                raw_hm = raw_hm.squeeze()
                inout = inout.cpu().detach().numpy()
                inout = 1 / (1 + np.exp(-inout))
                inout = (1 - inout) * 255
                norm_map = resize(raw_hm, (height, width)) - inout
                # print(f"norm_map: {norm_map}")

                # vis
                # plt.close()
                # fig = plt.figure()
                fig, ax = plt.subplots()  # generate figure with axes
                # fig.canvas.manager.window.move(0,0)
                plt.axis('off')
                plt.imshow(frame_raw)

                ax = plt.gca()
                rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2] - head_box[0],
                                         head_box[3] - head_box[1],
                                         linewidth=2, edgecolor=(0, 1, 0), facecolor='none')
                ax.add_patch(rect)

                if args.vis_mode == 'arrow':
                    if inout < args.out_threshold:  # in-frame gaze
                        pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                        pred_x_list.append(pred_x)
                        pred_y_list.append(pred_y)
                        face_list.append(face_status)

                        norm_p = [pred_x / output_resolution, pred_y / output_resolution]
                        gazex = norm_p[0] * width
                        gazey = norm_p[1] * height

                       # gaze_class = classify_gaze(result_folder, height, width, gazex, gazey)
                        facex = (head_box[0] + head_box[2]) / 2
                        facey = (head_box[1] + head_box[3]) / 2
                        gaze_class,is_extend,x_ext,y_ext= classify_gaze2(result_folder, height, width, gazex, gazey,facex,facey,image_in,face_status, folder)
                        circ = patches.Circle((gazex, gazey), height / 50.0, facecolor=(0, 1, 0), edgecolor='none')
                        ax.add_patch(circ)
                        if is_extend:
                            gazex=x_ext
                            gazey=y_ext
                        gaze_class_list.append(gaze_class)
                        # circ = patches.Circle((gazex, gazey), height / 50.0, facecolor=(0, 1, 0), edgecolor='none')
                        # ax.add_patch(circ)
                        plt.plot((gazex, (head_box[0] + head_box[2]) / 2),
                                 (gazey, (head_box[1] + head_box[3]) / 2), '-', color=(0, 1, 0, 1))
                        myFPS = 1.0 / (time.time() - start_fps)
                        plt.text(10, 0, '(class: {}, face: {}) FPS: {:.1f} '.format(get_classname(gaze_class),
                                                                                  face_status, myFPS), size=16,
                                 color=(0, 1, 0))
                    else:
                        pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                        pred_x_list.append(42)
                        pred_y_list.append(42)
                        face_list.append(face_status)

                        norm_p = [pred_x / output_resolution, pred_y / output_resolution]
                        gazex = norm_p[0] * width
                        gazey = norm_p[1] * height
                        circ = patches.Circle((gazex, gazey), height / 50.0, facecolor=(0, 1, 0), edgecolor='none')
                        ax.add_patch(circ)
                        plt.plot((gazex, (head_box[0] + head_box[2]) / 2),
                                 (gazey, (head_box[1] + head_box[3]) / 2), '-', color=(0, 1, 0, 1))
                        gaze_class_pred = 4
                        gaze_class = classify_gaze(result_folder, height, width, gazex, gazey)
                        gaze_class_list.append(gaze_class)

                        myFPS = 1.0 / (time.time() - start_fps)
                        plt.text(10, 0,
                                 '(pred: {} saved: {} face: {}) FPS: {:.1f}'.format(get_classname(
                                     gaze_class_pred), get_classname(gaze_class), face_status, myFPS), size=12,
                                 color=(0, 1, 0))
                        plt.text(10, 30, 'Gaze out of image', size=12, color=(0, 1, 0))
                else:
                    plt.imshow(norm_map, cmap='jet', alpha=0.2, vmin=0, vmax=255)

            else:
                fig, ax = plt.subplots()  # generate figure with axes
                plt.axis('off')
                plt.imshow(frame_raw)
                face_status = 0

                if args.vis_mode == 'arrow':
                    pred_x = 42
                    pred_y = 42
                    pred_x_list.append(pred_x)
                    pred_y_list.append(pred_y)
                    face_list.append(face_status)

                    norm_p = [pred_x / output_resolution, pred_y / output_resolution]
                    gazex = norm_p[0] * width
                    gazey = norm_p[1] * height
                    gaze_class_pred = classify_gaze(result_folder, height, width, gazex, gazey)
                    # circ = patches.Circle((gazex, gazey), height / 50.0, facecolor=(0, 1, 0), edgecolor='none')
                    # ax.add_patch(circ)
                    # plt.plot((gazex, 0), (gazey, 0), '-', color=(0, 1, 0, 1))
                    gaze_class = 4
                    gaze_class_list.append(gaze_class)

                    myFPS = 1.0 / (time.time() - start_fps)
                    plt.text(10, 0, '(pred: {} saved: {} face: {}) FPS: {:.1f} '.format(get_classname(
                                 gaze_class_pred), get_classname(gaze_class), face_status, myFPS), size=12, color=(0, 1, 0))
                else:
                    print('no image detected!')
                    myFPS = 1.0 / (time.time() - start_fps)
                    plt.text(10, 0, 'FPS: {:.1f} (Face: {})'.format(myFPS, face_status), size=16, color=(0, 1, 0))
                    plt.imshow(norm_map, cmap='jet', alpha=0.2, vmin=0, vmax=255)

            # plt.show(block=False)
            # plt.pause(0.001)

            plt.draw()
            # plt.show()
            num = int(os.path.splitext(i)[0])
            fig.savefig(os.path.join(plot_out_path, "%05d.jpg" % num))
            plt.close()

            dataframe = pd.DataFrame(
                data=np.concatenate(
                    [np.array(pred_x_list, ndmin=2), np.array(pred_y_list, ndmin=2),
                     np.array(face_list, ndmin=2), np.array(gaze_class_list, ndmin=2)]).T,
                columns=["yaw", "pitch", "face", "class"])
            csv_path = 'output/resultslin/' + result_folder
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)
            dataframe.to_csv(csv_path + folder + '.csv', index=False)

        # print('DONE!')
        print(f'{folder} is done!')


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    # print(args.vis_mode)
    # root = '/home/linlincheng/Documents/Projects/attention-target-detection-master/'

    # output
    output_frame_root = './output/processed_frames_lin_invi/'
    output_video_root = './output/processed_videos/'
    vis_2020_root = output_frame_root + args.vis_mode + '/fig_vis_bb/'
    invis_2020_root = output_frame_root + args.vis_mode + '/fig_invis_bb/'
    invis_2020_root_object = output_frame_root + args.vis_mode + '/fig_invis_obj/'
    vis_root_video = output_video_root + args.vis_mode + '/vid_vis/'
    invis_root_video = output_video_root + args.vis_mode + '/vid_invis/'

    # input
    frame_vis = './data/frame_vis/'
    frame_invis = './data/frame_invis_bb/'
    frame_invis_object = './data/frame_invis_object/'
    head_vis = './data/face_vis/'
    head_invis = './data/face_invis/'
    # vis_video_folder = './visible_bb/'
    # invis_video_folder = './invisible_bb/'
    # vis_video_list = [vid for vid in os.listdir(vis_video_folder) if (vid.endswith(".MP4") or vid.endswith(".mp4"))]
    # vis_video_list = sorted(vis_video_list)
    # invis_video_list = [vid for vid in os.listdir(invis_video_folder) if (vid.endswith(".MP4") or vid.endswith(".mp4"))]
    # invis_video_list = sorted(invis_video_list)

    if not os.path.exists(vis_2020_root):
        os.makedirs(vis_2020_root)
    if not os.path.exists(invis_2020_root):
        os.makedirs(invis_2020_root)
    if not os.path.exists(invis_2020_root_object):
        os.makedirs(invis_2020_root_object)
    if not os.path.exists(vis_root_video):
        os.makedirs(vis_root_video)
    if not os.path.exists(invis_root_video):
        os.makedirs(invis_root_video)

    # for folder in os.listdir(frame_vis):
    #     plot_out_path = os.path.join(vis_2020_root + folder)
    #     # if os.path.exists(plot_out_path):
    #     #     print(f'{folder} is passed...')
    #     #     continue
    #     # else:
    #     image_dir = os.path.join(frame_vis, folder)
    #     head_info = os.path.join(head_vis + folder + '.txt')
    #     result_folder = 'vis/'
    #     if not os.path.exists(plot_out_path):
    #         os.makedirs(plot_out_path)
    #     run()

    for folder in os.listdir(frame_invis):
        image_dir = os.path.join(frame_invis, folder)
        head_info = os.path.join(head_invis + folder + '.txt')
        result_folder = 'invis/'
        plot_out_path = os.path.join(invis_2020_root + folder)
        if not os.path.exists(plot_out_path):
            os.makedirs(plot_out_path)
        # if not os.path.exists(head_info):
        run()

    # for folder in os.listdir(frame_invis_object):
    #     image_dir = os.path.join(frame_invis_object, folder)
    #     head_info = os.path.join(head_invis + folder + '.txt')
    #     result_folder = 'object/'
    #     plot_out_path = os.path.join(invis_2020_root_object + folder)
    #     if not os.path.exists(plot_out_path):
    #         os.makedirs(plot_out_path)
    #     run()

    print("--- Complete excecution = %s seconds ---" % (time.time() - start_time))

    # for video in vis_video_list:
    #     filename = os.path.splitext(video)[0]
    #     print(f"filename: {filename}")
    #     video_name = filename + '.avi'  # video file name
    #
    #     vis_2020_out = vis_2020_root + filename + '/'
    #     video_name = vis_root_video + filename + '.avi'
    #
    #     if not os.path.exists(vis_2020_out):
    #         os.makedirs(vis_2020_out)
    #     if not os.path.exists(vis_root_video):
    #         os.makedirs(vis_root_video)
    #
    #     run()
    #
    # for video in invis_video_list:
    #     filename = os.path.splitext(video)[0]
    #     print(f"filename: {filename}")
    #     video_name = filename + '.avi'  # video file name
    #
    #     invis_2020_out = invis_2020_root + filename + '/'
    #     video_name = invis_root_video + filename + '.avi'
    #
    #     if not os.path.exists(invis_2020_out):
    #         os.makedirs(invis_2020_out)
    #     if not os.path.exists(invis_root_video):
    #         os.makedirs(invis_root_video)
    #
    #     run()

    # images = [img for img in os.listdir(frame_dir) if
    #           (img.endswith(".jpg") and img.startswith("0"))]
    # images = sorted(images)
    # print(images[0:5])
    #
    # frame = cv2.imread(os.path.join(frame_dir, images[0]))
    #
    # if frame.shape == (2160, 3840, 3):
    #     frame = cv2.resize(frame, (1920, 1080))
    # # print(frame.shape)
    #
    # height, width, layers = frame.shape
    # video = cv2.VideoWriter(video_name, 0, 15, (width, height))
    #
    # gaze_image = [img for img in os.listdir(plot_out_path) if
    #           (img.endswith(".jpg") and img.startswith("0"))]
    # for img in gaze_image:
    #     video.write(img)
    # cv2.destroyAllWindows()
    # video.release()
