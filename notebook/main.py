import numpy as np
import matplotlib.pyplot as plt  # библиотека Matplotlib для визуализации
import cv2
from pathlib import Path
import pypdfium2
import os

os.environ["USE_TORCH"] = "1"
from doctr.models import ocr_predictor
import pandas as pd
from pdf2image import convert_from_path
from nms import non_max_suppression
import onnxruntime as ort
import torch
from sklearn.cluster import KMeans

IMAGE_EXTENTIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp")


class Table:
    def __init__(self):
        self.arr_xpos = None
        self.arr_ypos = None

    def from_coords(self, list_coords):
        print(len(list_coords))
        # центры класстеров по x coorditane каждого разпознаного объекта
        self.arr_xpos = np.array([x["x_c"] for x in list_coords])
        self.arr_ypos = np.array([x["y_c"] for x in list_coords])
        # центры класстеров по х для объектов
        kmean = KMeans(3, n_init=1)
        kmean.fit(self.arr_xpos.reshape(-1, 1))
        self.centers = np.sort(kmean.cluster_centers_.T[0])

        lst_number, lst_X, lst_Y = [], [], []
        for item in list_coords:
            number_pos = np.argmin(
                [
                    abs(item["x_c"] - self.centers[0]),
                    abs(item["x_c"] - self.centers[1]),
                    abs(item["x_c"] - self.centers[2]),
                ]
            )
            if number_pos == 0:
                lst_number.append(item.copy())
            elif number_pos == 1:
                lst_X.append(item.copy())
            elif number_pos == 2:
                lst_Y.append(item.copy())
        print(f"Списки: {len(lst_number)} и {len(lst_X)} и {len(lst_Y)}")
        self.lst_number = lst_number
        self.lst_X = lst_X
        self.lst_Y = lst_Y
        # self.lst_number = self.connect_part_in_column(self.lst_number)

        self.X_before = lst_X.copy()
        self.Y_before = lst_Y.copy()

        # if len(self.lst_X) != len(self.lst_Y):
        #     self.lst_X = self.connect_part_in_column(self.lst_X)
        #     self.lst_Y = self.connect_part_in_column(self.lst_Y)

        self.lst_X = self.merge_paths_column_list(self.lst_X)
        self.lst_Y = self.merge_paths_column_list(self.lst_Y)
        print(f"Списки: {len(self.lst_X)} и {len(self.lst_Y)}")
        # проверка элементов to float
        # if len(self.lst_X) != len(self.lst_Y):
        #     for k,item in enumerate(self.lst_X):
        #         try:
        #             value = self.string_to_float(item['value'])
        #             self.lst_X[k]['value'] = value
        #         except:
        #             del self.lst_X[k]
        #     for k,item in enumerate(self.lst_Y):
        #         try:
        #             value = self.string_to_float(item['value'])
        #             self.lst_Y[k]['value'] = value
        #         except:
        #             del self.lst_Y[k]
        # else:
        #     for k, (x_item,y_item) in enumerate(zip(self.lst_X, self.lst_Y)):
        #         self.lst_X[k]['value'] = self.string_to_float(x_item['value'])
        #         self.lst_Y[k]['value'] = self.string_to_float(y_item['value'])

        print(f"Списки: {len(self.lst_X)} и {len(self.lst_Y)}")
        self._create_cells(self.lst_number, self.lst_X, self.lst_Y)
        self.arr_xpos = np.array(
            [x["x_c"] for x in list(self.lst_number + self.lst_X + self.lst_Y)]
        )
        self.arr_ypos = np.array(
            [x["y_c"] for x in list(self.lst_number + self.lst_X + self.lst_Y)]
        )
        return self.cells

    def _create_cells(self, lst_number, lst_X, lst_Y):
        if len(lst_X) == len(lst_Y):
            self.cells = [[None for j in range(3)] for i in range(len(lst_X))]
            for i, (x_v, y_v) in enumerate(zip(lst_X, lst_Y)):
                self.cells[i][0] = self.get_nearest_number(x_v)
                self.cells[i][1] = x_v
                self.cells[i][2] = y_v
        else:
            number_line = max(len(lst_X), len(lst_Y))
            self.cells = [[None for j in range(3)] for i in range(number_line)]
            if len(lst_X) > len(lst_Y):
                for k, item in enumerate(lst_X):
                    self.cells[k][0] = self.get_nearest_number(item)
                    self.cells[k][1] = item
                    self.cells[k][2] = self.get_nearest_item_in_table(item, lst_Y)
            else:
                for k, item in enumerate(lst_Y):
                    self.cells[k][0] = self.get_nearest_number(item)
                    self.cells[k][1] = self.get_nearest_item_in_table(item, lst_X)
                    self.cells[k][2] = item
        return self.cells

    @property
    def X_xposition(self):
        return np.array([a["x_c"] for a in self.lst_X])

    @property
    def Y_xposition(self):
        return np.array([a["x_c"] for a in self.lst_Y])

    @property
    def varience(self):
        yvar_X = np.array([x["x_c"] for x in self.lst_X]).var()
        yvar_Y = np.array([x["x_c"] for x in self.lst_Y]).var()
        return yvar_X, yvar_Y

    def get_nearest_number(self, x_value):
        """
        Получить номер(первый столбец) точки для текущей координаты X(второй столбец) по позиции в изображении x_c и y_c
        """
        yc = x_value["y_c"]
        ypos_number = np.array([x["y_c"] for x in self.lst_number])
        ypos_X = np.array([x["y_c"] for x in self.lst_X])
        cell_height = (
            ypos_X[1:] - ypos_X[:-1]
        ).mean()  # шаг между координатой по Y(высота ячейки в таблицe)

        delta = abs(ypos_number - yc)
        index = np.argmin(delta)
        if delta[index] <= cell_height * (0.9):
            return self.lst_number[index]
        else:
            return {
                "value": "Err",
                "x_c": np.array([x["x_c"] for x in self.lst_number]).mean(),
                "y_c": yc,
            }

    def get_nearest_item_in_table(self, x_value, list_coords):
        """
        Получить Y(3-й столбец) координату для текущей координаты X(второй столбец)
        """
        yc = x_value["y_c"]
        ypos_Y = np.array([x["y_c"] for x in list_coords])

        # ypos_Y = np.array([x['y_c'] for x in self.lst_Y])
        # ypos_X = np.array([x['y_c'] for x in self.lst_X])

        cell_height = (abs(ypos_Y[1:] - ypos_Y[:-1])).mean()
        delta = abs(ypos_Y - yc)
        index = np.argmin(delta)
        if delta[index] <= cell_height * (0.9):
            return list_coords[index]
        else:
            return {
                "value": "Err",
                "x_c": np.array([x["x_c"] for x in list_coords]).mean(),
                "y_c": yc,
            }

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
                r = abs(item["y_c"] - first_item["y_c"])
                if r < rmin:
                    rmin = r
                    second_item = item
            return second_item

        def connect_paths_val(first, second):
            val = first["value"].replace(" ", "").replace(".", "") + second[
                "value"
            ].replace(" ", "")
            return val

        arr_x = np.array([a["x_c"] for a in list_items])
        x_mean = arr_x.mean()
        x_var = arr_x.var()
        result_list = []
        visited = []
        min_length = np.array([len(x["value"]) for x in list_items]).mean()
        for item in list_items:
            if item in visited:
                continue
            r = abs(item["x_c"] - x_mean) ** 2
            cur_len = len(item["value"])
            if cur_len < min_length - 1:
                second_item = find_other_part(list_items, item)
                # print(item, second_item)
                visited.append(second_item)
                new_item = {
                    "value": connect_paths_val(item, second_item),
                    "x_c": x_mean,
                    "y_c": (item["y_c"] + second_item["y_c"]) * 0.5,
                }
                result_list.append(new_item)
            else:
                result_list.append(item)
        return result_list

    @property
    def frame(self):
        return pd.DataFrame(
            {
                "N": [c[0]["value"] for c in self.cells],
                "X": [c[1]["value"] for c in self.cells],
                "Y": [c[2]["value"] for c in self.cells],
            }
        )

    @staticmethod
    def string_to_float(string):
        if type(string) == float:
            return string
        result = string
        result = result.replace("\\", "")
        result = result.replace("/", "").replace("%", "")
        if result.count(".") > 1:
            parts = result.split(".")
            result = "".join(parts[:-1]) + "." + parts[-1]
        if result.count(",") > 1:
            parts = result.split(",")
            result = "".join(parts[:-1]) + "." + parts[-1]
        if result.count(",") == 1 and result.count(".") == 1:
            result = result.replace(",", ".")
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
        return float(result.replace(",", "."))

    def merge_paths_column_list(self, list_items):
        ycoords = np.array([x["y_c"] for x in list_items])
        y_diff = ycoords[1:] - ycoords[:-1]
        index = np.where(y_diff > y_diff.mean())[0]
        yc_lvl = ycoords[index]

        n = len(y_diff)
        if n < 5:
            mean_delta = y_diff.mean()
        else:
            mean_delta = np.sort(y_diff)[3:].mean()

        ymain_lvl = [ycoords[0]]
        for v in ycoords:
            delta = abs(v - ymain_lvl[-1])
            if delta < mean_delta * 0.3:
                continue
            else:
                ymain_lvl.append(v)

        items_on_lvl = {f"{i}": [] for i, item in enumerate(ymain_lvl)}
        for v in list_items:
            k = np.argmin(abs(np.array(ymain_lvl) - v["y_c"]))
            items_on_lvl[str(k)].append(v)

        new_items_list = []
        for k, items in items_on_lvl.items():
            if len(items) > 1:
                val = "".join([str(v["value"]) for v in items])
                val = val.replace("-", "")
                yc = np.array([v["y_c"] for v in items]).mean()
                xc = np.array([v["x_c"] for v in items]).mean()
                new_items_list.append({"value": val, "x_c": xc, "y_c": yc})
            else:
                new_items_list.append(items[0])
        return new_items_list


def letterbox_image(image, expected_size=(736, 736)):
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_img = np.full((eh, ew, 3), 128, dtype="uint8")
    # fill new image with the resized image and centered it
    new_img[
        (eh - nh) // 2 : (eh - nh) // 2 + nh, (ew - nw) // 2 : (ew - nw) // 2 + nw, :
    ] = image.copy()
    return new_img


def preprocess_image(image):
    if image.shape[2] == 4:
        # convert the image from RGBA2RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # image = letterbox_image(image)
    image = cv2.resize(image, (736, 736))
    image = image / image.max()
    image_norm = image.copy()
    batch = torch.tensor(image[np.newaxis, :], dtype=torch.float).permute(0, 3, 1, 2)
    return batch, image_norm


def process_directory(root):
    root = Path(root)
    print(os.getcwd())
    # model = YOLO('./best.pt')
    for p in Path("./").glob("*.onnx"):
        path2model = str(p)
        print(p)
    model = ort.InferenceSession(path2model)
    ocr = ocr_predictor(
        reco_arch="crnn_mobilenet_v3_large",
        pretrained=True,
        detect_language=True,
        detect_orientation=False,
    )

    result_path = Path("./results")
    if result_path.exists():
        pass
    else:
        result_path.mkdir()
    files = list(Path(root).rglob("*.pdf"))
    if len(files) == 0:
        files = ["1"]
    print(len(files))
    for pdf_file in files:
        try:
            images = convert_from_path(
                pdf_file,
                # poppler_path=r".\poppler-23.11.0\Library\bin"
                poppler_path="./poppler-23.11.0/Library/bin",
            )
        except Exception as e:
            images = list(Path(root).rglob("*"))
            images = list(
                filter(
                    lambda x: True if x.suffix in IMAGE_EXTENTIONS else False, images
                )
            )
            print(e)
        print(images)
        for image in images:
            if isinstance(image, Path):
                img = cv2.imread(str(image))
            else:
                img = np.array(image)

            batch, or_img = preprocess_image(img)
            h_orig, w_orig = img.shape[:2]
            h_crop, w_crop = or_img.shape[:2]
            h_rate = h_orig / h_crop
            w_rate = w_orig / w_crop
            outputs = model.run(None, {"images": batch.numpy()})  # Print Result

            # preds = model.predict(img, classes = [0]) Yolo

            predict = non_max_suppression(
                torch.tensor(outputs[0]), conf_thres=0.5, iou_thres=0.5
            )[0]
            _cls = predict[:, 5]
            table_predict = predict[_cls == 0]
            num_table = 0
            for t_pred in table_predict:
                x1, y1, x2, y2 = t_pred[:4]

                x1, y1, x2, y2 = (
                    int(x1 * w_rate),
                    int(y1 * h_rate),
                    int(x2 * w_rate),
                    int(y2 * h_rate),
                )
                x1, y1, x2, y2
                img_crop = img[y1:y2, x1:x2, :]

                # yolo
                # x1,y1, x2, y2 = b.xyxy.detach().cpu().numpy()[0]
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                img_crop = cv2.erode(img_crop, kernel=np.ones((2, 2), np.uint8))
                result = ocr([img_crop])
                pages = result.export()["pages"]
                blocks = pages[0]["blocks"]
                coords = []
                for b in blocks:
                    for l in b["lines"]:
                        for w in l["words"]:
                            p1 = np.array(w["geometry"][0])
                            p2 = np.array(w["geometry"][1])
                            centr = (p2 + p1) / 2
                            coords.append(
                                {"value": w["value"], "x_c": centr[0], "y_c": centr[1]}
                            )

                try:
                    table = Table()
                    cells = table.from_coords(coords)
                    if isinstance(image, np.ndarray):
                        table.frame.to_csv(
                            f"./results/{pdf_file.stem}_table_{num_table}.csv",
                            sep=";",
                            index=False,
                        )
                    else:
                        table.frame.to_csv(
                            f"./results/{image.stem}_table_{num_table}.csv",
                            sep=";",
                            index=False,
                        )
                    num_table += 1
                except Exception as e:
                    print("Неудалось", pdf_file, " ", e)


process_directory("/storage/reshetnikov/sber_table/dataset/val/")
#
# process_directory("/storage/reshetnikov/sber_table/notebook/val/")
