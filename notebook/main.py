import numpy as np
import matplotlib.pyplot as plt # библиотека Matplotlib для визуализации
import cv2
from pathlib import Path
from ultralytics import YOLO
import os
os.environ['USE_TORCH'] = '1'
from doctr.models import ocr_predictor
import pandas as pd
from pdf2image import convert_from_path

class Table:
    def __init__(self):
        self.arr_xpos = None
        self.arr_ypos = None
    
    def from_coords(self, list_coords):
        print(len(list_coords))
        #центры класстеров по x coorditane каждого разпознаного объекта
        self.arr_xpos = np.array([x['x_c'] for x in list_coords])
        self.arr_ypos = np.array([x['y_c'] for x in list_coords])
        #центры класстеров по х для объектов
        self.centers = np.array([self.arr_xpos.min(), self.arr_xpos.mean(), self.arr_xpos.max()])
        lst_number, lst_X, lst_Y = [], [], []
        for item in list_coords:
            number_pos = np.argmin([abs(item['x_c'] - self.centers[0]), 
                                    abs(item['x_c'] - self.centers[1]),
                                    abs(item['x_c'] - self.centers[2])])
            if number_pos == 0:
                lst_number.append(item)
            elif number_pos == 1:
                lst_X.append(item)
            elif number_pos == 2:
                lst_Y.append(item)
        print(f"Списки: {len(lst_number)} и {len(lst_X)} и {len(lst_Y)}")
        self.lst_number = lst_number
        self.lst_X = lst_X
        self.lst_Y = lst_Y
        # self.lst_number = self.connect_part_in_column(self.lst_number)
        if len(self.lst_X) != len(self.lst_Y):
            self.lst_X = self.connect_part_in_column(self.lst_X)
            self.lst_Y = self.connect_part_in_column(self.lst_Y)
        #проверка элементов to float
        if len(self.lst_X) != len(self.lst_Y):
            for k,item in enumerate(self.lst_X):
                try:
                    value = self.string_to_float(item['value'])
                    self.lst_X[k]['value'] = value
                except:
                    del self.lst_X[k]
            for k,item in enumerate(self.lst_Y):
                try:
                    value = self.string_to_float(item['value'])
                    self.lst_Y[k]['value'] = value
                except:
                    del lst_Y[k]
        else:
            for k, (x_item,y_item) in enumerate(zip(self.lst_X, self.lst_Y)):
                self.lst_X[k]['value'] = self.string_to_float(x_item['value'])
                self.lst_Y[k]['value'] = self.string_to_float(y_item['value'])
                
        print(f"Списки: {len(self.lst_X)} и {len(self.lst_Y)}")
        self._create_cells(self.lst_number, self.lst_X, self.lst_Y)
        self.arr_xpos = np.array([x['x_c'] for x in list(self.lst_number + self.lst_X + self.lst_Y)])
        self.arr_ypos = np.array([x['y_c'] for x in list(self.lst_number + self.lst_X + self.lst_Y)])
        return self.cells
    
    def _create_cells(self, lst_number, lst_X, lst_Y):
        number_line = len(lst_X)
        self.cells = [[None for j in range(3)] for i in range(number_line)]
        for i,(x_v, y_v) in enumerate(zip(lst_X, lst_Y)):
            self.cells[i][0] = self.get_nearest_number(x_v)
            self.cells[i][1] = x_v
            self.cells[i][2] = y_v
        return self.cells
    
    @property
    def X_xposition(self):
        return np.array([a['x_c'] for a in self.lst_X])
    
    @property    
    def Y_xposition(self):
        return np.array([a['x_c'] for a in self.lst_Y])
    
    @property
    def varience(self):
        yvar_X = np.array([x['x_c'] for x in self.lst_X]).var()
        yvar_Y = np.array([x['x_c'] for x in self.lst_Y]).var() 
        return yvar_X, yvar_Y
    
    def get_nearest_number(self, x_value):
        """
            Получить номер точки для текущей координаты X по позиции в изображении x_c и y_c 
        """
        yc = x_value['y_c']
        ypos_number = np.array([x['y_c'] for x in self.lst_number])
        ypos_X = np.array([x['y_c'] for x in self.lst_X])
        cell_height = (ypos_X[1:] - ypos_X[:-1]).mean()    #шаг между координатой по Y(высота ячейки в таблицe)
        
        delta = abs(ypos_number - yc)
        index = np.argmin(delta)
        if delta[index]<=cell_height*(0.9):
            return self.lst_number[index]
        else:
            return {'value': 'Err',
                    'x_c': np.array([x['x_c'] for x in self.lst_number]).mean(),
                    'y_c': yc}
        
    def connect_part_in_column(self, list_items):
        """
            Соединить части координат в столбце
        """
        def find_other_part(list_items, first_item):
            rmin = 1000
            second_item = None
            for item in list_items:
                if item == first_item:
                    continue
                r = abs(item['y_c'] - first_item['y_c'])
                if r<rmin:
                    rmin = r
                    second_item = item
            return second_item
        
        def connect_paths_val(first, second):
            val = first['value'].replace(" ","").replace(".","") + second['value'].replace(" ", "")
            return val
        
        arr_x = np.array([a['x_c'] for a in list_items])
        x_mean = arr_x.mean()
        x_var  = arr_x.var()
        result_list = []
        visited = []
        min_length = np.array([len(x['value']) for x in list_items]).mean()
        for item in list_items:
            if item in visited:
                continue
            r = abs(item['x_c'] - x_mean)**2
            cur_len = len(item['value'])
            if cur_len < min_length - 1:
                print(item['value'], abs(item['x_c'] - x_mean)**2)
                second_item = find_other_part(list_items, item)
                # print(item, second_item)
                visited.append(second_item)
                new_item = {'value':connect_paths_val(item, second_item),
                            'x_c'  :x_mean,
                            'y_c'  :(item['y_c'] + second_item['y_c'])*0.5}
                result_list.append(new_item)
            else:
                result_list.append(item)
        return result_list
    @property
    def frame(self):
        new_items = []
        for t in self.cells:
            if t[0] == None or t[1] == None or t[2] == None:
                continue
            else:
                new_items.append(t)
        frame = pd.DataFrame({"N":[c[0]['value'] for c in new_items], "X":[c[1]['value'] for c in new_items], "Y":[c[2]['value'] for c in new_items]})
        return frame
    
    @staticmethod
    def string_to_float(string):
    

        if type(string) == float:
            return string
        result = string
        result = result.replace("\\","")
        result = result.replace("/","").replace("%","")
        if result.count(".") > 1:
            parts = result.split(".")
            result = "".join(parts[:-1]) + "." + parts[-1]
        if result.count(",") > 1:
            parts = result.split(",")
            result = "".join(parts[:-1]) + "." + parts[-1]
        if result.count(",") == 1 and result.count(".") == 1:
            result = result.replace(",",".")
            parts = result.split(".")
            result = "".join(parts[:-1]) + "." + parts[-1]

        ans = []
        for s in result:
            if s in ",.0123456789":
                ans.append(s)
        result = "".join(ans)    
        if len(result) == 1:
            if result not in "0123456789":
                return -1
        if len(result) == 0:
            return 0
        return float(result.replace(',','.'))
        
def process_directory(root):
    root = Path(root)
    model = YOLO('./runs/detect/train3/weights/best.pt')
    ocr = ocr_predictor(pretrained=True, detect_language=True, detect_orientation = False)


    files = list(Path(root).rglob('*.pdf'))
    print(len(files))
    for pdf_file in files:
        images = convert_from_path(pdf_file)
        for image in images:
            # img = cv2.imread(str(path))
            img = np.array(image)
            # img = cv2.resize(img, (740,740))
            preds = model.predict(img, classes = [0])   
            num_table = 0 
            for b in preds[0].boxes:
                if b.conf > 0.6:
                    x1,y1, x2, y2 = b.xyxy.detach().cpu().numpy()[0]    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    img_crop = img[y1:y2,x1:x2,:]
                    img_crop = cv2.erode(img_crop,kernel = np.ones((2,2), np.uint8))
                    result = ocr([img_crop])
                    pages = result.export()['pages']
                    blocks = pages[0]['blocks']
                    coords = []
                    for b in blocks:
                        for l in b['lines']:
                            for w in l['words']:
                                p1 = np.array(w['geometry'][0])
                                p2 = np.array(w['geometry'][1])
                                centr = (p2 + p1)/2
                                coords.append({'value':w['value'],'x_c':centr[0],'y_c':centr[1]})
                    # table = Table()
                    # cells = table.from_coords(coords)

                    # table.frame.to_csv(f'./results/{path.stem}_table_{num_table}.csv', sep=";", index=False)
                    # num_table+=1
                    try:                        
                        table = Table()
                        cells = table.from_coords(coords)

                        table.frame.to_csv(f'./results/{path.stem}_table_{num_table}.csv', sep=";", index=False)
                        num_table+=1
                    except:
                        print("Неудалось", path)
process_directory("/storage/reshetnikov/sber_table/dataset/tabl/230830032/")