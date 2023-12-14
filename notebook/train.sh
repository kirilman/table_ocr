yolo train model="yolov8m.pt" data="/storage/reshetnikov/sber_table/dataset/config.yaml" \
imgsz=720 batch=16 epochs=100 cache="ram" save device='cuda:2,3' 
