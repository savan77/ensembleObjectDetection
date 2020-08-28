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
    if 'mask_rcnn' in args.models:
        maskRcnn = testTimeAugmentation.MaskRCNNPred('/mnt/weights/mask_rcnn_coco.h5', '/mnt/weights/coco.names')
        listModels.append(maskRcnn)
    if 'retinanet' in args.models:
        retinaResnet50 = testTimeAugmentation.RetinaNetResnet50Pred('/mnt/weights/resnet50_coco_best_v2.1.0.h5', '/mnt/weights/coco.csv')
        listModels.append(retinaResnet50)
        
#     listaModels = [retinaResnet50, maskRcnn]
    models(listModels,args.images_path,args.option)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default="retinanet")
    parser.add_argument('--images_path', default='/data/images')
    parser.add_argument('--option', default='unanimous')
    args = parser.parse_args()
    main(args)
