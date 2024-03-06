from chardet import detect
from matplotlib.pyplot import grid
from tools.dataset_converters.textdet.naf_converter import collect_annotations
import torch
import cv2
import numpy as np
#import supervision
from mmocr.apis.inferencers import mmocr_inferencer
from mmocr.models.common.backbones.unet import DeconvModule
from mmocr.utils.polygon_utils import poly2bbox
from mmocr.apis.inferencers import MMOCRInferencer
def Recognition(image):
    #set the device
    DEVICE = torch.device('cuda"11.8' if torch.cuda.is_available() else 'cpu')

    #assign mode config path and model weight path
    detection_config_path = 'configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py'
    detection_weight_path = 'weightss/DBNet/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth'

    recognition_config_path = 'configs/textrecog/abinet/abinet_20e_st-an_mj.py'
    recognition_weight_path = 'weightss/ABINet/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth'

    #call mmocr inference
    mmocr_inferencer = MMOCRInferencer(
        det=detection_config_path,
        det_weights= detection_weight_path,
        rec=recognition_config_path,
        rec_weights=recognition_weight_path,
        device=DEVICE
    )

    #read the image
    image_bgr = cv2.imread(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_image = image_bgr
    #resize image
    image_rgb = cv2.resize(image_rgb, (1024,1024))
    image_bgr = cv2.resize(image_bgr, (1024,1024))

    #get mmocr inference result
    result = mmocr_inferencer(image_rgb)['predictions'][0]
    recognized_text = result['rec_texts']
    detected_polygons = result['det_polygons']


    detected_boxes = torch.tensor(
        np.array([poly2bbox(poly) for poly in detected_polygons]),
        device=DEVICE
    )

    detected_boxes = np.array(detected_boxes)

    counter = 0
    boxes = []
    words = []
    target = []
    row = 0
    #Group the recognition result
    for bbox in detected_boxes:
        x1,y1, x2, y2 = bbox
        col = []
        arr = []
        if len(boxes) == 0:
            col = [int(x1),int(y1),int(x2),int(y2)]
            boxes.append(col)
            words.append(recognized_text[counter])
            arr.append(recognized_text[counter])
            arr.append(col)
            target.append(arr)
        else:
            row = len(boxes) - 1
            height = 0
            if(int(y1)<int(boxes[row][1])):
                height = int(boxes[row][1]) - int(y1)
            else:
                height = int(y1) - int(boxes[row][1])
            if (height<20):
                if(int(boxes[row][2])<int(x2)):
                    words[row] = words[row]+" "+recognized_text[counter]
                    boxes[row][2] = int(x2)
                else:
                    word = words[row].split()
                    words[row] = " ".join(word[:-1]) + " "+recognized_text[counter]+" "+ word[-1]

                boxes[row][3] = int(y2)
            else:
                col = [int(x1),int(y1),int(x2),int(y2)]
                boxes.append(col)
                arr.append(recognized_text[counter])
                arr.append(col)
                target.append(arr)
                words.append(recognized_text[counter])
        counter+=1
    #Extra the book name
    result = ""
    for k in range(len(boxes)):
        x1 = int(boxes[k][0])
        y1 = int(boxes[k][1])
        x2 = int(boxes[k][2])
        y2 = int(boxes[k][3])
        height = abs(int(y1)-int(y2))
        if(height>45):
            width = abs(int(target[k][1][0])-(target[k][1][2]))
            chart = len(target[k][0])
            #print(width)
            #print(chart)
            #print(target[k][0])
            if((int(width)/int(chart))>50):
                if(result!=""):
                    result=words[k]+" "+result
                else:
                    result=words[k]
                cv2.rectangle(image_bgr,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(image_bgr, words[k],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    #display image
    scale_percent = 0.5
    width = int(image_bgr.shape[1] * scale_percent)
    height = int(image_bgr.shape[0] * scale_percent)
    dim = (width, height)
    resized_image = cv2.resize(image_bgr, dim, interpolation=cv2.INTER_AREA)
    print("file path: "+image+"; Book Name: "+result)
    #cv2.imshow(image,resized_image)
    #cv2.waitKey(0)
    return result

filename = 'book1.jpg'
image = 'demo\\book2.jpg'
result = Recognition(image)
print(result)