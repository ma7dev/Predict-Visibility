import numpy as np
import cv2
import sys, csv
import random
import operator
from matplotlib import pyplot as plt

# target files
SOURCE_DIR = "./data/"
#FOLDERS = ["MOT17-02","MOT17-04","MOT17-05","MOT17-09","MOT17-10","MOT17-11","MOT17-13", "CVPR19-01", "CVPR19-02", "CVPR19-03", "CVPR19-05", "ETH-Jelmoli", "ETH-Seq01", "KITTI16", "KITTI19", "PETS09-S2L2", "TUD-Crossing"]
#FOLDERS = ["MOT17-02","MOT17-04","MOT17-05","MOT17-09","MOT17-10","MOT17-11","MOT17-13"]
FOLDERS = ["MOT17-04"]

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
    seq_imgs = []
    for i in range(1,seq_length+1):
        curr_path = path
        if(i < 10):
            curr_path += "00000"+str(i)+im_ext
        elif(10 <= i <100):
            curr_path += "0000"+str(i)+im_ext
        elif(100 <= i <1000):
            curr_path += "000"+str(i)+im_ext
        elif(1000 <= i <10000):
            curr_path += "00"+str(i)+im_ext
        elif(10000 <= i <100000):
            curr_path += "0"+str(i)+im_ext
        elif(100000 <= i <1000000):
            curr_path += str(i)+im_ext
        print(curr_path)
        seq_imgs.append(cv2.cvtColor(cv2.imread(curr_path), cv2.COLOR_BGR2RGB))
    # convert to numpy array for easy modification
    seq_imgs = np.array(seq_imgs)
    #print(seq_imgs.shape)
    #resolution = [seq_imgs.shape[2], seq_imgs.shape[1]]
    resolution = [im_width, im_height]
    # read gt.txt file
    # convert from csv (having comma) to a list
    gt = csv.reader(open(gt_file),delimiter=',')
    # sorted by frames (asend.) the list based on the frame order
    sorted_gt = sorted(gt, key=lambda row: int(row[0]), reverse=False)
    return seq_imgs, resolution, sorted_gt


def display(sorted_gt, seq_imgs, resolution, frame_stop = -1, stop=False):
    # curr_frame is used to label all objects in a frame, once curr_frame doesn't match with current frame (when reading), it means all objects in curr_frame has been visited
    #print(curr_frame)
    # read every line in gt.tx
    curr_frame = int(sorted_gt[0][0])-1
    for line in sorted_gt:
        # frame, id (label in my case), x, y, w, h, class, visibility
        frame = int(line[0])-1
        if(curr_frame != frame and frame_stop == curr_frame):
            print("FRAME:",curr_frame)
            plt.imshow(seq_imgs[curr_frame])
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        if(stop):
            break
        label = str(line[1])
        x = int(line[2])
        y = int(line[3])
        w = int(line[4])
        h = int(line[5])
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.rectangle(seq_imgs[frame], (x1, y1), (x2, y2), color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        x2_label = x1 + t_size[0] + 3
        y2_label = y1 + t_size[1] + 4
        cv2.rectangle(seq_imgs[frame], (x1, y1), (x2_label, y2_label), color, -1)
        cv2.putText(seq_imgs[frame], label, (x1, y2_label), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        curr_frame = frame

def checkFailed(sorted_gt, seq_imgs, resolution, stop=False):
    # curr_frame is used to label all objects in a frame, once curr_frame doesn't match with current frame (when reading), it means all objects in curr_frame has been visited
    #print(curr_frame)
    # read every line in gt.tx
    curr_frame = int(sorted_gt[0][0])-1
    for line in sorted_gt:
        # frame, id (label in my case), x, y, w, h, class, visibility
        frame = int(line[0])-1
        if(curr_frame != frame):
            print("FRAME:",curr_frame)
            plt.imshow(seq_imgs[curr_frame])
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        if(stop):
            break
        label = str(line[1])
        x = int(line[2])
        y = int(line[3])
        w = int(line[4])
        h = int(line[5])
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.rectangle(seq_imgs[frame], (x1, y1), (x2, y2), color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        x2_label = x1 + t_size[0] + 3
        y2_label = y1 + t_size[1] + 4
        cv2.rectangle(seq_imgs[frame], (x1, y1), (x2_label, y2_label), color, -1)
        cv2.putText(seq_imgs[frame], label, (x1, y2_label), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        curr_frame = frame


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
    penalty = background_object_area - getArea(getOverlappedPosition(background_object, [-1, 0, 0, resolution[0], resolution[1], -1., -1, -1.]))
    vis_area = 0
    if (2 in set_):
        vis_area = set_[2]
    predicted_vis = float((vis_area)/background_object_area)
    return predicted_vis
def approxVis(folder_name, sorted_gt, seq_imgs, resolution, save=True, stop=False):
    # read every line in gt.txt
    objects_predicted = []
    frames_predicted = []
    objects_truth = []
    frames_truth = []
    # read every example in the frame
    # curr_frame is used to label all objects in a frame, once curr_frame doesn't match with current frame (when reading), it means all objects in curr_frame has been visited
    curr_frame = int(sorted_gt[0][0])-1
    for line in sorted_gt:
        # frame, id (label in my case), x, y, w, h, conf, class, visibility
        frame = int(line[0])-1
        label = str(line[1])
        x = int(line[2])
        y = int(line[3])
        w = int(line[4])
        h = int(line[5])
        conf = float(line[6])
        c = int(line[7])
        vis = float(line[8])
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h
        if curr_frame == frame:
            # label, x1, y1, x2, y2, conf, c, vis
            objects_predicted.append([label, x1, y1, x2, y2, conf, c, -1.])
            objects_truth.append([label, x1, y1, x2, y2, conf, c, vis])
        else:
            frames_predicted.append(sorted(objects_predicted, key=lambda row: row[4], reverse=False))
            frames_truth.append(sorted(objects_truth, key=lambda row: row[4], reverse=False))
            if(stop):
                break
            curr_frame = frame
            objects_predicted = []
            objects_truth = []
            objects_predicted.append([label, x1, y1, x2, y2, conf, c, -1.])
            objects_truth.append([label, x1, y1, x2, y2, conf, c, vis])

    failed_tests_in_a_folder = []
    # calculate visibility
    #print("frame, id(label), x1, y1, x2, y2, conf., c, vis, predicted_vis")
    failed_tests = []
    failed_tests_counter = 0
    for i, frame in enumerate(frames_predicted):
        #print(len(frame))
        #print("----------------------------------------------")
        #print("-FRAME:", i)
        for j, background_object in enumerate(frame):
            #print("++++++++++++++++++++++++++++++++++++++++++++")
            #print("--BACKGROUND:", background_object[0])
            #print("++++++++++++++++++++++++++++++++++++++++++++")
            # to evaluate the targeted class
            if(background_object[6] in [1, 2, 7]):
                predicted_vis = getPredictedVis(frame, background_object, i, j, resolution)
                if(abs(predicted_vis - frames_truth[i][j][7]) > 0.05):
                    failed_tests_counter += abs(predicted_vis - frames_truth[i][j][7])
                    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    #print("--ID:", background_object[0])
                    #print("--CLASS:", background_object[6])
                    #print("background_object_area - overlapped_positions_area + interesection_of_overlapped_positions_area - penalty")
                    #print(background_object_area, overlapped_positions_area, intersection_of_overlapped_positions_area, penalty)
                    #print("--(", background_object[1],",",background_object[2],") vs (" , frames_truth[i][j][1],",",frames_truth[i][j][2],")")
                    #print("----PREDICTED:", predicted_vis)
                    #print("----ACTUAL:", frames_truth[i][j][7])
                    #print(i, background_object[0], background_object[1], background_object[2], background_object[3], background_object[4], background_object[5], background_object[6], frames_truth[i][j][7], predicted_vis)
                    failed_tests.append([i, background_object[0], background_object[1], background_object[2], background_object[3], background_object[4], background_object[5], background_object[6], frames_truth[i][j][7], predicted_vis])
    if(save):
        # write into a file
        with open(folder_name+".csv", mode="w") as target_file:
            target_writer = csv.writer(target_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for failed_test in failed_tests:
                target_writer.writerow([
                        str(failed_test[0]),
                        str(failed_test[1]),
                        str(failed_test[2]),
                        str(failed_test[3]),
                        str(failed_test[4]),
                        str(failed_test[5]),
                        str(failed_test[6]),
                        str(failed_test[7]),
                        str(failed_test[8]),
                        str(failed_test[9])
                       ] )
        target_file.close()
    failed_tests_in_a_folder = round(float(len(failed_tests)/len(sorted_gt)),2)
    failed_tests_average = 0
    if(len(failed_tests) > 0):
        failed_tests_average = float(failed_tests_counter / len(failed_tests))
    return failed_tests_in_a_folder, failed_tests_average

def main():
    failed_tests_across_folders = []
    failed_tests_across_folders_average = []
    for folder_name in FOLDERS:
        seq_imgs, resolution, sorted_gt = readFiles(folder_name)
        #frame_stop = -1
        frame_stop = 831
        display(sorted_gt, np.copy(seq_imgs),resolution,frame_stop)
        #failed_tests_in_a_folder, failed_tests_average = approxVis(folder_name, sorted_gt, np.copy(seq_imgs), resolution)
        #failed_tests_in_a_folder, failed_tests_average = approxVis(folder_name, sorted_gt, np.copy(seq_imgs), resolution, save=False)
        failed_tests_across_folders.append(failed_tests_in_a_folder)
        failed_tests_across_folders_average.append(failed_tests_average)
    plt.bar(FOLDERS, failed_tests_across_folders)
    plt.show()
    plt.bar(FOLDERS, failed_tests_across_folders_average)
    plt.show()
if __name__ == "__main__":
    main()
