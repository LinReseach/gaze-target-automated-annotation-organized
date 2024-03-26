import cv2
import os

from face_detection import RetinaFace
from skimage import io

# /home/linlincheng/Documents/Projects/L2CS-Net-main/input/00016.jpg
# /home/linlincheng/Documents/Projects/attention-target-detection-master/data/demo/Proefpersoon_63001_video

# file_list_all = ["33003_1", "MVI_0163", "Proefpersoon11024_sessie2", "Proefpersoon11026_sessie2",
#         "Proefpersoon11044_sessie1_uitleg", "Proefpersoon51015_Sessie1", "Proefpersoon_63009_video_1",
# "Proefpersoon_63009_video_2"]

# file_list = ["Proefpersoon11024_sessie2", "Proefpersoon11026_sessie2",
#         "Proefpersoon11044_sessie1_uitleg", "Proefpersoon51015_Sessie1", "Proefpersoon_63009_video_1",
#         "Proefpersoon_63009_video_2"]
# root = 'data/'
# folder_name = '51007_sessie2_taskrobotEngagement'
# mydir = root + folder_name + '/'

# file0 = '00000.jpg'     # Box: [611.258   175.973   743.6726  328.65482]
# fileb = '00000_bb.jpg'     # Box:
# filef = '00000_filled.jpg'     # Box:
# file1 = '00016.jpg'  # Box: [1185.2328      3.131538 1377.7991    229.38211 ]
# file2 = '02431.jpg'  # Box: [1276.4504   217.70497 1459.562    439.61298]
# file11 = '02640.jpg'  # Box: [1103.0753     24.435362 1288.2554    256.2012  ]
# file22 = '02658.jpg'  # Box: [1133.0159     78.317184 1329.2452    321.1303  ]
# file33 = '02730.jpg'  # Box: [1194.7106   107.36793 1401.7561   338.77078]
# file34 = '00034.jpg'  # Box: [2098.0422  837.823  2446.031  1318.399 ]
# file35 = '00035.jpg'  # Box: [2112.609   843.3309 2459.9548 1306.2451]
# file38 = '00038.jpg'  # Box: [2096.1758  817.3677 2468.9224 1305.0012]
# file55 = '00055.jpg'  # Box: [2326.7014  725.8587 2743.1719 1223.5688][left, top, right, bottom]
# file56 = '00056.jpg'  # Box: [2436.3647  707.6366 2877.001  1223.5104]
# file57 = '00057.jpg'  # Box: [2634.8755  693.6042 3072.272  1179.206 ]
# image = root + file0
# """


vis_dir = './data/frame_vis/'
invis_dir = './data/frame_invis/'

# print(os.listdir(vis_dir))

for folder in os.listdir(vis_dir):
    img_folder = vis_dir + folder + '/'
    images = [img for img in os.listdir(img_folder) if (img.endswith(".jpg") and img.startswith("0"))]
    images = sorted(images)

    vis_face_info = folder + '.txt'
    detector = RetinaFace(gpu_id=0)  # 0 for gpu, -1 for CPU

    with open(os.path.join('./data/face_vis/' + vis_face_info), 'w') as f:
        for image in images:
            # print(image)
            f.write(image)
            image = cv2.imread(os.path.join(img_folder, image))
            faces = detector(image)
            try:
                box, landmarks, score = faces[0]
                f.write(',')
                temp_list = []
                for b in box:
                    temp_b = round(float(b))
                    temp_list.append(temp_b)
                    f.write(','.join(str(b) for b in temp_list))
                    # f.write(str(temp_b))
                    # f.write(',')
                f.write('\n')
            except IndexError:
                box = ['None', 'None', 'None', 'None']
                f.write(',')
                f.write(','.join(str(b) for b in box))
                f.write('\n')
    print(f"{vis_face_info} is done!")
print("All visible cases are done!")

for folder in os.listdir(invis_dir):
    img_folder = invis_dir + folder + '/'
    images = [img for img in os.listdir(img_folder) if (img.endswith(".jpg") and img.startswith("0"))]
    images = sorted(images)

    invis_face_info = folder + '.txt'
    detector = RetinaFace(gpu_id=0)  # 0 for gpu, -1 for CPU

    with open(os.path.join('./data/face_invis/' + invis_face_info), 'w') as f:
        for image in images:
            # print(image)
            f.write(image)
            image = cv2.imread(os.path.join(img_folder, image))
            faces = detector(image)
            try:
                box, landmarks, score = faces[0]
                f.write(',')
                temp_list = []
                for b in box:
                    temp_b = round(float(b))
                    temp_list.append(temp_b)
                    f.write(','.join(str(b) for b in temp_list))
                    # f.write(str(tem_b))
                    # f.write(',')
                f.write('\n')
            except IndexError:
                box = ['None', 'None', 'None', 'None']
                f.write(',')
                f.write(','.join(str(b) for b in box))
                f.write('\n')
    print(f"{invis_face_info} is done!")
print("All invisible cases are done!")

"""
images = [img for img in os.listdir(mydir) if
          (img.endswith(".jpg") and img.startswith("0"))]
images = sorted(images)
print(images[0:5])
# print(image[-5:])

frame = cv2.imread(os.path.join(mydir, images[0]))

# info_list = []

vis_face_info = './data/frame_vis/' + folder_name + '.txt'
invis_face_info = './data/frame_invis/' + folder_name + '.txt'
detector = RetinaFace(gpu_id=0)  # 0 for gpu, -1 for CPU

with open(face_info, 'w') as f:
    for image in images:
        # print(image)
        f.write(image)
        # image = cv2.imread(os.path.join(image_folder, "frame" + str(image) + ".jpg"))
        # print(image[2:])
        image = cv2.imread(os.path.join(mydir, image))
        # info_list.append(image)
        # print(info_list)
        face = detector(image)
        box, landmarks, score = face[0]
        f.write(',')
        f.write(','.join(str(b) for b in box))
        f.write('\n')
    print("Done!")"""
# """

# dataframe = pd.DataFrame(
#     data=np.concatenate([np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2)]).T,
#     columns=["yaw", "pitch"])
# dataframe.to_csv(output_file_name, index=False)

# img = io.imread('/home/linlincheng/Documents/Projects/L2CS-Net-main/input/00038.jpg')
"""
# to print single image information
detector = RetinaFace(gpu_id=0)  # 0 for gpu, -1 for CPU
img = cv2.imread(image)
# info_list.append(image)
# print(info_list)
face = detector(img)
box, landmarks, score = face[0]

if face is not None:
    print("Face Not None!")
    # variables used in case of single person eye tracking
    biggest_width = 0
    biggest_height = 0
    x_min_ = 0
    x_max_ = 0
    y_min_ = 0
    y_max_ = 0
    box, landmarks, score = face[0]
    print(type(box))
    print(type(landmarks))
    print(type(score))
    print(f"Box: {box}")
    print(landmarks)
    print(score)"""
"""
for box, landmarks, score in face:
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
    # The following if allow us to save just the values of the biggest square (single-person gaze tracking)
    # Demo_pepper for
    if bbox_width > biggest_width:
        biggest_width = bbox_width
        biggest_height = bbox_height
        x_min_ = x_min
        x_max_ = x_max
        y_min_ = y_min
        y_max_ = y_max
        print(f"x_min is: {x_min_}")
        print(f"x_max is: {x_max_}")
        print(f"y_min is: {y_min_}")
        print(f"y_max is: {y_max_}")
    # x_min = max(0,x_min-int(0.2*bbox_height))
    # y_min = max(0,y_min-int(0.2*bbox_width))
    # x_max = x_max+int(0.2*bbox_height)
    # y_max = y_max+int(0.2*bbox_width)
    # bbox_width = x_max - x_min
    # bbox_height = y_max - y_min
else:
    print("Face is None")

# put inside the for loop for eye tracking of all the faces in the frame (now it's just the bigger one)
# Crop image
# img = frame[y_min_:y_max_, x_min_:x_max_]
#
# # img = cv2.resize(img, (224, 224))
# try:
#     img = cv2.resize(img, (224, 224))
# except Exception as e:
#     # return None, None, None
#     print(str(e))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# im_pil = Image.fromarray(img)
# img = transformations(im_pil)
# img = Variable(img).cuda(gpu)
# img = img.unsqueeze(0)

# if frame.shape == (2160, 3840, 3):
#     frame = cv2.resize(frame, (1920, 1080))
#     cv2.rectangle(frame, (int(x_min_ / 2), int(y_min_ / 2)), (int(x_max_ / 2), int(y_max_ / 2)), (0, 255, 0), 1)
#
# cv2.rectangle(frame, (x_min_, y_min_), (x_max_, y_max_), (0, 255, 0), 1)

# cv2.imshow("Demo", frame)
# if cv2.waitKey(1) & 0xFF == 27:
#    break
# success, frame = cap.read()
# cap.release()
# cv2.destroyAllWindows()"""

"""
if __name__ == '__main__':

    # processing from local
    # image_folder = 'input/4k_exp_setting'
    image_folder = 'input/'
    # participant = 'p2'
    outputname = 'output/fdtest'
    video_name = outputname + 'simple_mytest6_30.avi'
    output_file_name = outputname + 'simple_mytest6_30.csv'
    # video_name = outputname + 'simple_calibration_4k.avi'
    # output_file_name = outputname + 'simple_calibration_4k.csv'
    # Transformation needed after the face detection

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    arch = args.arch
    snapshot_path = args.snapshot

    model = getArch(arch, 90)  # 28 for MPIIGaze, 90 for Gaze360
    # model = torch.nn.DataParallel(model)  # only for MPII
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)  # , map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)

    # processing with local images
    # images = [int(img[5:-4]) for img in os.listdir(image_folder) if img.endswith(".jpg")] #remove 'frame' and order images
    # images = [f'%(number)05d' % {"number": int(img[5:-4])} for img in os.listdir(image_folder) if img.endswith(".jpg")] #remove 'frame' and order images

    # rename all the images
    # print(len(os.listdir(image_folder)))
    # count = 1
    # for img in os.listdir(image_folder):
    # images = f'%(number)05d' % {"number": int(img[5:-4])

    # images = int(float(img[5:-4]))
    # images = sorted(images)
    # print("old: "+img)
    # print("new: "+str(images))
    # os.rename(image_folder + "/" + img, image_folder + "/" + str('%(number)05d' % {"number": images}) + ".jpg")
    # count+=1
    # print("count: "+str(count))

    images = [img for img in os.listdir(image_folder) if
              (img.endswith(".jpg") and img.startswith("0"))]  # from pepper experiment
    images = sorted(images)
    print(images[0:5])
    # print(image[-5:])

    # frame = cv2.imread(os.path.join(image_folder, "frame" + str(images[0]) + ".jpg"))
    frame = cv2.imread(os.path.join(image_folder, images[0]))  # from pepper experiment

    if frame.shape == (2160, 3840, 3):
        frame = cv2.resize(frame, (1920, 1080))
    # print(frame.shape)

    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    # we save pitch and yaw into a pandas DataFrame
    pitch_predicted_ = []
    yaw_predicted_ = []
    for image in images:
        # print(image)
        # image = cv2.imread(os.path.join(image_folder, "frame" + str(image) + ".jpg"))
        # print(image[2:])
        image = cv2.imread(os.path.join(image_folder, image))
        img, pitch_predicted, yaw_predicted = prediction(transformations, model, image)
        # img = prediction(transformations, model, image)
        pitch_predicted_.append(pitch_predicted)
        yaw_predicted_.append(yaw_predicted)
        video.write(img)

    dataframe = pd.DataFrame(
        data=np.concatenate([np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2)]).T,
        columns=["yaw", "pitch"])
    dataframe.to_csv(output_file_name, index=False)
    cv2.destroyAllWindows()
    video.release()"""""
