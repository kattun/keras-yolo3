import sys
import os
import pandas as pd
import argparse
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo, score_file):
    df = pd.DataFrame(columns=['ImageID', 'PredictionString'])
    for i, img_name in enumerate(os.listdir(yolo.input)):
        try:
            image = Image.open(os.path.join(yolo.input, img_name))
        except:
            print('Open Error!')
            print(img_name)
            continue
        else:
            out_boxes, out_scores, out_classes, r_image = yolo.detect_image(image)

            # save image
            if not yolo.noimgdet:
                r_image.save(os.path.join(yolo.output, img_name))

            pred_str = ""
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = yolo.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                pred_str += " {} {} {} {} {} {}".format(predicted_class, score, top, left, bottom, right)

            # save score
            imageid = os.path.splitext(os.path.basename(img_name))[0]
            s = pd.Series([imageid, pred_str], index=df.columns)
            df = df.append(s, ignore_index=True)

        if i % 100 == 0: df.to_csv(score_file, index=False, header=True)

    yolo.close_session()
    df.to_csv(score_file, index=False, header=True)

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_image',
        help = "Images input dir"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, required=False,default="output",
        help = "[Optional] Images output dir"
    )

    parser.add_argument(
        "--score_file", nargs='?', type=str, required=False,default="submission.csv",
        help = "[Optional] score file name"
    )

    parser.add_argument(
        "--noimgout", nargs='?', type=bool, required=False,default=True,
        help = "[Optional] no output detected image"
    )

    FLAGS = parser.parse_args()

    flags_var = vars(FLAGS)
    print("Image detection mode")


    if "input" in FLAGS:
        yolo = YOLO(**flags_var)
        os.makedirs(yolo.output, exist_ok=True)
        detect_img(yolo, flags_var["score_file"])
    else:
        print("Must specify at least image_input_path.  See usage with --help.")

