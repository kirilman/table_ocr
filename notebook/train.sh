yolo train model="/storage/reshetnikov/sber_table/notebook/runs/detect/train11/weights/best.pt" \
data="/storage/reshetnikov/sber_table/dataset/table_v2/config.yaml" imgsz=1024 batch=8 epochs=100 cache="ram" \
device="cuda:2,3" degrees = 4 scale = 0.2 fliplr = 0.3 flipud=0.2 
# name = "after_train"


# yolo train model="yolov8l.pt" data="/storage/reshetnikov/sber_table/dataset/tndr_set/fold/Fold_0/config.yaml" \
# imgsz=1024 batch=16 epochs=100 save device="cuda:2,3" degrees = 4 scale = 0.2 fliplr = 0.3 flipud=0.2 cache="ram" 
