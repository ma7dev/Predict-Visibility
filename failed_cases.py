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
FOLDERS = ["MOT17-05"]

def readFiles(folder_name):
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
            curr_path += "00000"+str(i)+".jpg"
        elif(10 <= i <100):
            curr_path += "0000"+str(i)+".jpg"
        elif(100 <= i <1000):
            curr_path += "000"+str(i)+".jpg"
        elif(1000 <= i <10000):
            curr_path += "00"+str(i)+".jpg"
        elif(10000 <= i <100000):
            curr_path += "0"+str(i)+".jpg"
        elif(100000 <= i <1000000):
            curr_path += str(i)+".jpg"
        print(curr_path)
        seq_imgs.append(cv2.cvtColor(cv2.imread(curr_path), cv2.COLOR_BGR2RGB))
    # convert to numpy array for easy modification
    seq_imgs = np.array(seq_imgs)
    #print(seq_imgs.shape)
    #resolution = [seq_imgs.shape[2], seq_imgs.shape[1]]
    resolution = [im_width, im_height]
    # read failed tests
    failed_tests = []
    with open(folder_name+'.csv', 'r') as target_file:
        reader = csv.reader(target_file)
        failed_tests = list(reader)
    target_file.close()

    return seq_imgs, resolution, failed_tests


def display(failed_tests, seq_imgs, resolution, stop=False):
    # curr_frame is used to label all objects in a frame, once curr_frame doesn't match with current frame (when reading), it means all objects in curr_frame has been visited
    #print(curr_frame)
    # read every line in gt.tx
    curr_frame = int(failed_tests[0][0])
    for line in failed_tests:
        # frame, id (label in my case), x1, y1, x2, y2, conf., class, visibility, predicted visibility
        frame = int(line[0])
        if(curr_frame != frame):
            plt.imshow(seq_imgs[curr_frame])
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        if(stop):
            break
        label = str(line[1])
        x1 = int(line[2])
        y1 = int(line[3])
        x2 = int(line[4])
        y2 = int(line[5])
        conf = float(line[6])
        c = int(line[7])
        vis = float(line[8])
        predicted_vis = float(line[9])
        print("FRAME:",frame,"ID:",label,"PREDICTED:",predicted_vis,"ACTUAL:",vis)
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.rectangle(seq_imgs[frame], (x1, y1), (x2, y2), color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        x2_label = x1 + t_size[0] + 3
        y2_label = y1 + t_size[1] + 4
        cv2.rectangle(seq_imgs[frame], (x1, y1), (x2_label, y2_label), color, -1)
        cv2.putText(seq_imgs[frame], label, (x1, y2_label), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        curr_frame = frame

def main():
    for folder_name in FOLDERS:
        seq_imgs, resolution, failed_tests = readFiles(folder_name)
        display(failed_tests, np.copy(seq_imgs),resolution)
if __name__ == "__main__":
    main()
