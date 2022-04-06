# from paddleocr import PaddleOCR,draw_ocr
import cv2
import numpy as np
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import torch
import tqdm
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from shapely.geometry import Polygon
import pyclipper
import cv2

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    # dictionary = 'aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ'
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D‑", "d‑"]

def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition

def decode_recognition(rec):
    CTLABELS = [" ","!",'"',"#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","{","|","}","~","ˋ","ˊ","﹒","ˀ","˜","ˇ","ˆ","˒","‑",]
    s = ""
    for c in rec:
        c = int(c)
        if c < 104:
            s += CTLABELS[c]
        elif c == 104:
            s += u"口"
    return decoder(s)

def get_mini_boxes(contour, thr):
    bounding_box = cv2.minAreaRect(contour)
    bounding_box = list(bounding_box)
    bounding_box[1] = list(bounding_box[1])
    if bounding_box[2]<=-45:
        bounding_box[1][1] = bounding_box[1][1]*thr
    else:
        bounding_box[1][0] = bounding_box[1][0]*thr
    bounding_box[1] = tuple(bounding_box[1])
    bounding_box = tuple(bounding_box)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box

def get_mini_boxes_1(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val in value:
            return key

def merge_boxes(boxes, recs, trh):
    dict_bbox = {}
    x=0
    for i in range(len(boxes)-2):
        tmp_box = [i]
        db_copy1 = dict_bbox.copy()
        for key, value in db_copy1.items():
            if i in value:
                tmp_box = db_copy1[key]
                del dict_bbox[key]
                break
        for j in range(i+1, len(boxes)-1):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > trh:
                db_copy = dict_bbox.copy()
                check = False
                for key, value in db_copy.items():
                    if i in value:
                        check = True
                        tmp_box.extend(db_copy[key])
                        del dict_bbox[key]
                        break
                if check == False:
                    tmp_box.append(j)
        dict_bbox[x] = tmp_box
        x+=1
    recs_out = []
    for key, value in dict_bbox.items():
        tmp_str = []
        for i in value:
            # tmp_str += ' ' + recs[i]
            # print(boxes[i])
            tmp_str.append([recs[i], int((boxes[i][0][0]+ boxes[i][2][0])/2)])
            # tmp_str += ' ' + decode_recognition(recs[i])
            # print(decode_recognition(recs[i]))
        recs_out.append(tmp_str)
    return dict_bbox, recs_out

def combine(dict_box, h, w, boxes):
    bboxs = []
    for key, db in dict_box.items():
        list_box = []
        for j in db:
            list_box.append(boxes[j])
        h1 = h
        h2 = 0
        h3 = 0
        h4 = h
        w1 = w
        w2 = w
        w3 = 0
        w4 = 0
        for box in list_box:
            if box[0,0] < h1:
                h1 = box[0,0]
            if box[1,0] > h2:
                h2 = box[1,0]
            if box[2,0] > h3:
                h3 = box[2,0]
            if box[3,0] < h4:
                h4 = box[3,0]
            if box[0,1] < w1:
                # p1 = [box[0,0], box[0,1]]
                w1 = box[0,1]
            if box[1,1] < w2:
                w2 = box[1,1]
            if box[2,1] > w3:
                # p3 = [box[2,0], box[2,1]]
                w3 = box[2,1]
            if box[3,1] > w4:
                w4 = box[3,1]                       
            tmp = np.array([[h1,w1],[h2,w2],[h3,w3],[h4,w4]])
            # print("tmp", tmp)
        bboxs.append(tmp.astype(np.int16))
    return bboxs

def rec_to_str(recs):
    rec_out = []
    for i in recs:
        i =  sorted(i, key=lambda x: x[1])
        i = " ".join(item[0] for item in i)
        rec_out.append(i)
    return rec_out

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--inputfile", nargs="+", help="A list of array of segmentation")
    parser.add_argument("--output", nargs="+", help="A list of array of segmentation")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

import shapely
from shapely.geometry import Point, Polygon

def scale_points(mask):
    mask_tmp = mask.copy()
    for i in range(2,len(mask_tmp)-2):
        for j in range(2,len(mask_tmp[i])-2):
            if mask_tmp[i][j] != 0:
                mask[i-2][j-2] = mask[i-2][j-1] = mask[i-2][j] = mask[i-2][j+1] = mask[i-2][j+2] = mask[i-1][j-2] = mask[i-1][j-1] = mask[i-1][j] = mask[i-1][j+1] = mask[i-1][j+2] = mask[i][j-2] = mask[i][j-1] = mask[i][j+1] = mask[i][j+2] = mask[i+1][j-2] = mask[i+1][j-1] = mask[i+1][j] = mask[i+1][j+1] = mask[i+1][j+2] = mask[i+2][j-2] = mask[i+2][j-1] = mask[i+2][j] = mask[i+2][j+1] = mask[i+2][j+2] = mask_tmp[i][j]
    return mask

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    path_segment = ''
    if args.inputfile:
        path_segment = args.inputfile[0]
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            print(path)
            txt_name = str(path.split("/")[-1].split(".")[0]) + '.txt'
            txt_file = os.path.join(path_segment, txt_name)
            img = read_image(path, format="BGR")
            h, w, _ = img.shape
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            mask = np.loadtxt(txt_file,  dtype=np.float32)
            # mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
            mask = scale_points(mask)
            # mask1 = scale_points(mask)
            # mask2 = scale_points(mask1)

            # kernel = np.ones((5, 5), np.uint8)
            # Using cv2.erode() method 
            # erode0 = cv2.erode(mask2, kernel) 
            # edges = mask2 - erode0
            # mask_blur = cv2.GaussianBlur(mask2, (3,3), 0)
            # sobelx = cv2.Sobel(src=mask_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            
            outs = cv2.findContours((mask* 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # print(outs)
            if len(outs) == 3:
                img, contours, _ = outs[0], outs[1], outs[2]
            elif len(outs) == 2:
                contours, _ = outs[0], outs[1]

            box_sign = []
            for contour in contours:
                points = get_mini_boxes_1(contour)
                points = np.array(points)
                box_sign.append(points)

            # bnr_name = str(path.split("/")[-1].split(".")[0]) + '.jpg'
            # bnr_file = os.path.join("output_test/bnr_img", bnr_name)
            # cv2.imwrite(bnr_file, mask * 255)

            # bs_name = str(path.split("/")[-1].split(".")[0]) + '.txt'
            # bs_file = os.path.join("output_test/bs_output", txt_name)
            # bs_out = open(bs_file, 'w+', encoding='utf-8')
            # for i in box_sign:
            #     np.savetxt(bs_out, i)
            #     bs_out.write("\n")

            dict_box_sign = {}
            dict_box_sign_out = {}
            dict_rec_sign = {}
            dict_rec_sign_out = {}
            in_signboard = 0
            full_box = 0
            img_draw = cv2.imread(path)
            for i in range(len(box_sign)):
                dict_box_sign[i] = []
                dict_box_sign_out[i] = []
                dict_rec_sign[i] = []
                dict_rec_sign_out[i] = []
            if "instances" in predictions:
                predictions_cpu = predictions["instances"].to(torch.device("cpu"))
                beziers = predictions_cpu.beziers.numpy()
                recs = predictions_cpu.recs

                for bezier, rec in zip(beziers, recs):
                    bezier = bezier.reshape(-1,1,2).astype(int)
                    bounding_box = cv2.minAreaRect(bezier)
                    midpoint = Point(bounding_box[0])
                    for i in range(len(box_sign)):
                        poly = Polygon(box_sign[i])
                        if midpoint.within(poly):
                            in_signboard+=1
                            dict_box_sign[i].append(bezier)
                            dict_rec_sign[i].append(decode_recognition(rec))
                    full_box = len(beziers)
                for i in range(len(dict_box_sign)):
                    boxes = []
                    reces = []
                    # print(dict_box_sign[i])
                    for bezier, rec in zip(dict_box_sign[i], dict_rec_sign[i]):
                        unclip_ratio = 1.0
                        bezier = bezier.reshape(-1,1,2)
                        points = get_mini_boxes(bezier, 1.8)
                        box = np.array(points, dtype=np.int16)

                        box[:, 0] = np.clip(np.round(box[:, 0]), 0, w)
                        box[:, 1] = np.clip(np.round(box[:, 1]), 0, h)

                        boxes.append(box.astype(np.int16))
                        reces.append(rec)

                    dict_box, rec_out = merge_boxes(boxes, reces, 0.1)
                    # print(rec_out)
                    rec_outs = rec_to_str(rec_out)
                    # print(rec_outs)
                    bboxs = combine(dict_box, h, w, boxes)

                    # dict_box, rec_out = merge_boxes(bboxs, rec_out, 0.1)
                    # rec_outs = rec_to_str(rec_out)
                    # bboxs = combine(dict_box, h, w, bboxs)

                    # dict_boxs, rec_outs = merge_boxes(bboxs, rec_out, 0.1)
                    # bboxes = combine(dict_boxs, h, w, bboxs)
                    # rec_outs = rec_to_str(rec_outs)

                    dict_box_sign_out[i] = bboxs
                    dict_rec_sign_out[i] = rec_outs
                    # print(len(bboxes))
                    # print(len(rec_outs))
            txt_name = str(path.split("/")[-1].split(".")[0]) + '.txt'
            if args.output: 
                output_path = os.path.join(args.output[0], txt_name)
                output_file = open(output_path, 'w+', encoding='utf-8')
                output_file.write(str(in_signboard) + " ")
                output_file.write(str(full_box) + '\n')
                output_file.write(str(dict_box_sign_out))
                output_file.write(str(dict_rec_sign_out))
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )