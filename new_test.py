import numpy as np
import cv2
import random
import operator
from matplotlib import pyplot as plt
import csv


# target files
SOURCE_DIR = "./data/"
FOLDERS = ["AVG-TownCentre", "ETH-Jelmoli", "ETH-Seq01", "KITTI16", "KITTI19", "PETS09-S2L2", "TUD-Crossing"]
#FOLDERS = ["MOT17-04"]

def readFiles(folder_name):
    gt_file = SOURCE_DIR+folder_name+"/gt/gt.txt"
    seqinfo_file = SOURCE_DIR+folder_name+"/seqinfo.ini"
    frame_rate = seq_length = im_width = im_height = -1
    im_dir = im_ext = ""
    fp = open(seqinfo_file)
    for i, line in enumerate(fp):
        if i == 2:
            im_dir = str(line.split("=")[1]).rstrip()
        elif i == 3:
            frame_rate = int(line.split("=")[1])
        elif i == 4:
            seq_length = int(line.split("=")[1])
            #seq_length = 2
        elif i == 5:
            im_width = int(line.split("=")[1])
        elif i == 6:
            im_height = int(line.split("=")[1])
        elif i == 7:
            im_ext = str(line.split("=")[1]).rstrip()
    fp.close()
    path = SOURCE_DIR+folder_name+"/"+im_dir+"/"
    print("path:",path)
    # read sequence of imgs
    #resolution = [seq_imgs.shape[2], seq_imgs.shape[1]]
    resolution = [im_width, im_height]
    # read gt.txt file
    # convert from csv (having comma) to a list
    gt = csv.reader(open(gt_file),delimiter=',')
    # sorted by frames (asend.) the list based on the frame order
    sorted_gt = sorted(gt, key=lambda row: float(row[0]), reverse=False)
    return resolution, sorted_gt


# testing

# test = p. vs p.
# matched frames (just to see)
#   plot


# (x1, y1)--------------+
#    |                  |
#    |                  |
#    |                  |
#    |                  |
#    +---------------(x2, y2)
# 1=x1, 2=y1, 3=x2, 4=y2
# improve to IOU
def doOverlap(obj_1, obj_2):
#    return not (obj_1[3] <= obj_2[1] or  # left
#                obj_1[4] <= obj_2[2] or  # bottom
#                obj_1[1] >= obj_2[3] or  # right
#                obj_1[2] >= obj_2[4])    # top
    if obj_1[1] > obj_2[3] or obj_1[3] < obj_2[1]:
        return False
    if obj_1[2] > obj_2[4] or obj_1[4] < obj_2[2]:
        return False
    return True

def getOverlappedPosition(obj_1, obj_2):
    label = -1
    x1 = max(obj_1[1], obj_2[1])
    y1 = max(obj_1[2], obj_2[2])
    x2 = min(obj_1[3], obj_2[3])
    y2 = min(obj_1[4], obj_2[4])
    conf = -1.
    c = -1
    vis = -1.
    return [label, x1, y1, x2, y2, conf, c, vis]

def getArea(obj):
    #print(obj)
    w = float(obj[3] - obj[1])
    l = float(obj[4] - obj[2])
    area = w*l
    return area

def setToOnes(matrix_map, background_object, overlapped_object, resolution):
    for i in range(overlapped_object[2]-background_object[2], overlapped_object[4]-background_object[2]):
        for j in range(overlapped_object[1]-background_object[1], overlapped_object[3]-background_object[1]):
            matrix_map[i][j] = 1
    return matrix_map

def getPredictedVis(frame, background_object, i, j, resolution):
    overlapped_positions = []
    # get all of the overlapped areas
    matrix_map = np.zeros((background_object[4]-background_object[2], background_object[3]-background_object[1]), dtype=int)
    #print(matrix_map.shape)
    for j_foreground in range(0,len(frame)):
        if(background_object[4] <= frame[j_foreground][4] and j != j_foreground):
            if(doOverlap(background_object, frame[j_foreground])):
                #print("---FOREGROUND:", frame[j_foreground],"===== YES")
                overlapped_position = getOverlappedPosition(background_object, frame[j_foreground])
                overlapped_positions.append(overlapped_position)
    for overlapped_position in overlapped_positions:
        matrix_map = setToOnes(matrix_map, background_object, overlapped_position, resolution)
    vis_in_frame = getOverlappedPosition(background_object, [-1, 0, 0, resolution[0], resolution[1], -1., -1, -1.])

    for i_frame in range(vis_in_frame[2]-background_object[2], vis_in_frame[4]-background_object[2]):
        for j_frame in range(vis_in_frame[1]-background_object[1], vis_in_frame[3]-background_object[1]):
            if (matrix_map[i_frame][j_frame] == 0):
                matrix_map[i_frame][j_frame] = 2
    unique, counts = np.unique(matrix_map, return_counts=True)
    set_ = dict(zip(unique, counts))
    background_object_area = getArea(background_object)
    vis_area = 0
    if (2 in set_):
        vis_area = set_[2]
    predicted_vis = float((vis_area)/background_object_area)
    return predicted_vis
def approxVis(folder_name, sorted_gt, resolution, save=True, stop=False):
    # read every line in gt.txt
    objects_predicted = []
    frames_predicted = []
    # read every example in the frame
    # curr_frame is used to label all objects in a frame, once curr_frame doesn't match with current frame (when reading), it means all objects in curr_frame has been visited
    curr_frame = round(float(sorted_gt[0][0]))-1
    for line in sorted_gt:
        print(line)
        # frame, id (label in my case), x, y, w, h, conf, 3d_x, 3d_y, 3d_z
        frame = round(float(line[0]))-1
        label = round(float(line[1]))
        x = round(float(line[2]))
        y = round(float(line[3]))
        w = float(line[4])
        h = float(line[5])
        conf = 1.
        x_3d = 0.
        y_3d = 0.
        z_3d = 0.
        if(len(line) > 6):
            conf = float(line[6])
        if(len(line) > 7):
            x_3d = float(line[7])
            y_3d = float(line[8])
            z_3d = float(line[9])
        x1 = x
        y1 = y
        x2 = x1 + int(round(w))
        y2 = y1 + int(round(h))
        if curr_frame == frame:
            # label, x1, y1, x2, y2, conf, c, vis, w, l, 3d_x, 3d_y, 3d_z
            objects_predicted.append([label, x1, y1, x2, y2, conf, 1, -1., w, h, x_3d, y_3d, z_3d, frame+1])
        else:
            frames_predicted.append(sorted(objects_predicted, key=lambda row: row[4], reverse=False))
            if(stop):
                break
            curr_frame = frame
            objects_predicted = []
            objects_predicted.append([label, x1, y1, x2, y2, conf, 1, -1., w, h, x_3d, y_3d, z_3d, frame+1])
    frames_predicted.append(sorted(objects_predicted, key=lambda row: row[4], reverse=False))

    # calculate visibility
    #print("frame, id(label), x1, y1, x2, y2, conf., c, vis, predicted_vis")
    tests = []
    for i, frame in enumerate(frames_predicted):
        #print(len(frame))
        #print("----------------------------------------------")
        #print("-FRAME:", i)
        for j, background_object in enumerate(frame):
            #print("++++++++++++++++++++++++++++++++++++++++++++")
            #print("--BACKGROUND:", background_object[0])
            #print("++++++++++++++++++++++++++++++++++++++++++++")
            # to evaluate the targeted class
            predicted_vis = getPredictedVis(frame, background_object, i, j, resolution)
            tests.append([background_object[13], background_object[0], background_object[1], background_object[2], background_object[8], background_object[9], background_object[5], background_object[6], predicted_vis, background_object[10], background_object[11], background_object[12]])
    # write into a file
    with open(folder_name+"_gt.txt", mode="w") as target_file:
        target_writer = csv.writer(target_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for test in tests:
            target_writer.writerow([
                    str(test[0]),
                    str(test[1]),
                    str(test[2]),
                    str(test[3]),
                    str(test[4]),
                    str(test[5]),
                    str(test[6]),
                    str(test[7]),
                    str(test[8]),
                    str(test[9]),
                    str(test[10]),
                    str(test[11])
                   ] )
    target_file.close()

def main():
    for folder_name in FOLDERS:
        resolution, sorted_gt = readFiles(folder_name)
        #frame_stop = -1
        frame_stop = 831
        #display(sorted_gt, np.copy(seq_imgs),resolution,frame_stop)
        approxVis(folder_name, sorted_gt, resolution)
        #failed_tests_in_a_folder, failed_tests_average = approxVis(folder_name, sorted_gt, np.copy(seq_imgs), resolution, save=False)
if __name__ == "__main__":
    main()
