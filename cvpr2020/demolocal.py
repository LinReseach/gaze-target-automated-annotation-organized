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
        print(img_list[0:5])
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
                        circ = patches.Circle((gazex, gazey), height / 50.0, facecolor=(0, 1, 0), edgecolor='none')
                        ax.add_patch(circ)
                        plt.plot((gazex, (head_box[0] + head_box[2]) / 2),
                                 (gazey, (head_box[1] + head_box[3]) / 2), '-', color=(0, 1, 0, 1))
                        gaze_class = classify_gaze(result_folder, height, width, gazex, gazey)
                        gaze_class_list.append(gaze_class)

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
            csv_path = 'output/results/' + result_folder
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
    output_frame_root = './output/processed_frames/'
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
