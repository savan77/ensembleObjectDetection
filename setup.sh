git clone https://github.com/fizyr/keras-retinanet
pip3 install ./keras-retinanet/ --user
git clone https://github.com/onepanelio/Mask_RCNN
pip3 install ./Mask_RCNN/ --user
wget https://www.dropbox.com/sh/n21kckhsi200b52/AABxspis34aAZiMUp_cQ6RYFa?dl=1 -O weights.zip
unzip weights.zip -d /mnt/weights/
pip3 install -r requirements.txt
