from signboard_detect import inference_signboard
import os
import numpy as np
import argparse
import tqdm
import cv2

def get_parser():
    parser = argparse.ArgumentParser(description="Signboard Detection")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--output", nargs="+", help="A list of array of segmentation")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    
    for path in tqdm.tqdm(args.input):
        print(path)
        img = cv2.imread(path)
        dimensions = img.shape
        hei,wid = dimensions[0], dimensions[1]
        print(hei,wid)
        segment_array = inference_signboard(path).astype(int)
        h, w = segment_array.shape
        print(h, w)
        if hei == h and wid == w:
            segment_array = segment_array
        else:
            segment_array = cv2.rotate(segment_array, cv2.cv2.ROTATE_90_CLOCKWISE)
        txt_name = str(path.split("/")[-1].split(".")[0]) + '.txt'
        if args.output: 
            output_path = os.path.join(args.output[0], txt_name)

            np.savetxt(output_path, segment_array)
