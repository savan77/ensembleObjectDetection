import testTimeAugmentation
import function
import os
import shutil
import argparse
import ensembleOptions
from mainModel import models
from imutils import paths
import argparse
import warnings
warnings.filterwarnings('ignore')



def main(args):
    listModels = []
    models_list = args.models.split(",")
    print("Models to be run: ", models_list)
    if 'mask_rcnn' in models_list:
        maskRcnn = testTimeAugmentation.MaskRCNNPred('/mnt/weights/mask_rcnn_coco.h5', '/mnt/weights/coco.names')
        listModels.append(maskRcnn)
    if 'retinanet' in models_list:
        retinaResnet50 = testTimeAugmentation.RetinaNetResnet50Pred('/mnt/weights/resnet50_coco_best_v2.1.0.h5', '/mnt/weights/coco.csv')
        listModels.append(retinaResnet50)
    if 'yolo_darknet' in models_list:
        yoloDarknet = testTimeAugmentation.DarknetYoloPred('/mnt/weights/yolov3.weights', '/mnt/weights/coco.names','/mnt/weights/yolov3.cfg')
        listModels.append(yoloDarknet)
        
#     listaModels = [retinaResnet50, maskRcnn]
    models(listModels,args.images_path,args.option, args.combine)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default="retinanet,yolo_darknet")
    parser.add_argument('--images_path', default='/data/images')
    parser.add_argument('--option', default='unanimous')
    parser.add_argument("--combine", default=False)
    args = parser.parse_args()
    main(args)
