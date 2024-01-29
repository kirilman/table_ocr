import numpy as np
import matplotlib.pyplot as plt  # библиотека Matplotlib для визуализации
import cv2
from pathlib import Path
import os

os.environ["USE_TORCH"] = "1"
from doctr.models import ocr_predictor
import pandas as pd
from nms import non_max_suppression
import onnxruntime as ort
import torch
from sklearn.cluster import KMeans

# from brisque.brisque import BRISQUE
from piq import brisque

try:
    import fitz
except Exception as er:
    print(er)

from PIL import Image
from pdf2image import convert_from_path

IMAGE_EXTENTIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp")

import warnings

warnings.filterwarnings("ignore")


def isfloat(s):
    try:
        float(s)
        return True
    except:
        return False


class Table:
    def __init__(self, model, image):
        self.arr_xpos = None
        self.arr_ypos = None
        self.model = model
        self.image = image

    def check_centers(self):
        if abs(self.centers[1] - self.centers[0]) / self.centers[1] * 100 < 20:
            print("Координаты для кластера '№ номер' и 'X' совпали")
            self.centers[0] = 0.01
        if abs(self.centers[1] - self.centers[2]) / self.centers[1] * 100 < 20:
            print("Координаты для кластера 'X' и 'Y' совпали")
            self.centers[1] = self.centers[0]
            self.centers[0] = 0.001
        d1 = abs(self.centers[1] - self.centers[0])
        d2 = abs(self.centers[2] - self.centers[1])
        if d1 / d2 < 0.22:
            print("Отношение сторон")
            if d1 < d2:
                self.centers[0] = 0.001
            else:
                self.centers[2] = 1
                self.centers[1] = (self.centers[0] + self.centers[1]) / 2
                self.centers[0] = 0.001

    def from_coords(self, list_coords):
        list_float = []
        for x in list_coords:
            try:
                if float(x["value"].replace(",", ".")):
                    list_float.append(x)
            except:
                pass
        # центры класстеров по x coorditane каждого разпознаного объекта
        if len(list_float) > 2:
            self.arr_xpos = np.array([x["x_c"] for x in list_float])
            self.arr_ypos = np.array([x["y_c"] for x in list_float])
        else:
            self.arr_xpos = np.array([x["x_c"] for x in list_coords])
            self.arr_ypos = np.array([x["y_c"] for x in list_coords])

        # центры класстеров по х для объектов
        cluster_centers = np.array(
            [
                [
                    self.arr_xpos.min(),
                    self.arr_xpos.mean(),
                    np.sort(self.arr_xpos)[len(self.arr_xpos) // 2 :].max(),
                ]
            ]
        ).T
        kmean = KMeans(init=cluster_centers, n_clusters=3, n_init=1)
        kmean.fit(self.arr_xpos.reshape(-1, 1))
        self.centers = np.sort(kmean.cluster_centers_.T[0])
        # print(self.centers, (self.centers[1] - self.centers[0]) / self.centers[1])
        self.check_centers()

        lst_number, lst_X, lst_Y = [], [], []
        for item in list_coords:
            number_pos = np.argmin(
                [
                    abs(item["x_c"] - self.centers[0]),
                    abs((item["x_c"]) - self.centers[1]),
                    abs((item["x_c"] + item["p_right"][0]) / 2 - self.centers[2]),
                ]
            )
            if number_pos == 0:
                lst_number.append(item.copy())
            elif number_pos == 1:
                lst_X.append(item.copy())
            elif number_pos == 2:
                lst_Y.append(item.copy())
        # print(f"Списки: {len(lst_number)} и {len(lst_X)} и {len(lst_Y)}")
        self.lst_number = sorted(lst_number, key=lambda d: d["y_c"])
        lst_X = sorted(lst_X, key=lambda d: d["y_c"])
        lst_Y = sorted(lst_Y, key=lambda d: d["y_c"])

        self.lst_X = lst_X
        self.lst_Y = lst_Y
        # self.lst_number = self.connect_part_in_column(self.lst_number)
        # сортировка списков по координате y
        self.X_before = lst_X.copy()
        self.Y_before = lst_Y.copy()

        self.lst_X = self.merge_paths_column_list(self.lst_X)
        self.lst_Y = self.merge_paths_column_list(self.lst_Y)

        self._create_cells(self.lst_number, self.lst_X, self.lst_Y)

        # to float

        for cell in self.cells:
            cell[1]["value"] = self.string_to_float(cell[1]["value"])
            cell[2]["value"] = self.string_to_float(cell[2]["value"])

        self.arr_xpos = np.array(
            [x["x_c"] for x in list(self.lst_number + self.lst_X + self.lst_Y)]
        )
        self.arr_ypos = np.array(
            [x["y_c"] for x in list(self.lst_number + self.lst_X + self.lst_Y)]
        )
        return self.cells

    def _create_cells(self, lst_number, lst_X, lst_Y):
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
        if len(ypos_number) == 0:
            return {
                "value": "Err",
                "x_c": self.centers[0],
                "y_c": yc,
            }

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
        df = pd.DataFrame(
            {
                # "N": [c[0]["value"] for c in self.cells],
                "X": [c[1]["value"] for c in self.cells],
                "Y": [c[2]["value"] for c in self.cells],
            }
        )
        # drop item
        df = df[
            df.X.apply(
                lambda x: True
                if len(str(x)) > 5 and x > 250000 and x < 5000000
                else False
            )
        ]
        df = df[df.Y.apply(lambda x: True if len(str(x)) > 5 else False)]
        return df

    @staticmethod
    def string_to_float(string):
        if type(string) == float:
            return string
        if type(string) == int:
            return string

        result = string
        result = result.replace("/", "7")
        result = result.replace("B", "3")
        result = result.replace("\\", "")
        result = result.replace("%", "")
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
        # for y in ycoords:
        #     print(y)
        y_diff = ycoords[1:] - ycoords[:-1]
        index = np.where(y_diff > y_diff.mean())[0]
        yc_lvl = ycoords[index]

        n = len(y_diff)
        if n < 5:
            mean_delta = y_diff.mean()
        else:
            mean_delta = np.sort(y_diff)[2:-2].mean()

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
                # if len(items) > 2:
                #     print("Три пробела в цифре ", items)
                # Соединить несколько боксов на уровне
                x_left = np.array([v["p_left"][0] for v in items])
                x_right = np.array([v["p_right"][0] for v in items])

                y_left = np.array([v["p_left"][1] for v in items])
                y_right = np.array([v["p_right"][1] for v in items])

                shape = self.image.shape
                h, w = shape[0], shape[1]
                r = abs((x_left.max() - x_left.min())) * 0.01
                x1 = int((x_left.min() - r) * w)
                y1 = int(y_left.min() * h)
                x2 = int((x_right.max() + r) * w)
                y2 = int(y_right.max() * h)
                img = self.image[y1:y2, x1:x2]
                result = self.model.reco_predictor([img])
                val = result[0][0]

                def merge_parths_value(parth_items):
                    list_values = []
                    for p in sorted(parth_items, key=lambda d: d["x_c"]):
                        val = p["value"]
                        if len(val) > 2:
                            list_values.append(val)
                        else:
                            if val.isdigit():
                                list_values.append(val)
                            else:
                                print("Не цифра ", val)
                    return "".join(list_values)

                # new_val = merge_parths_value(items)
                # val = "".join(
                #     [str(v["value"]) for v in sorted(items, key=lambda d: d["x_c"])]
                # )

                # val = val.replace("-", "")
                yc = np.array([v["y_c"] for v in items]).mean()
                xc = np.array([v["x_c"] for v in items]).mean()
                new_items_list.append({"value": val, "x_c": xc, "y_c": yc})
            else:
                m = items[0]
                if 4 < len(m["value"]) <= 13:
                    new_items_list.append(items[0])
        # выкинуть выбрасы по X
        # for x in new_items_list:
        #     print(x)
        return new_items_list


def letterxyxy2org(x1, y1, x2, y2, h_org, w_org, h_let, w_let, pad_h, pad_w, scale):
    """
    Конверт координаты бокса x1y1x2y2 из lettr изображения в x1y1x2y2 исходного изображения
    """
    xa1 = int(x1 / w_let * w_org - pad_w / scale)
    ya1 = int(y1 / h_let * h_org - pad_h / scale)

    xa2 = int(x2 / w_let * w_org - pad_w / scale)
    ya2 = int(y2 / h_let * h_org - pad_h / scale)
    return xa1, ya1, xa2, ya2


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
    step_h = (eh - nh) // 2
    step_w = (ew - nw) // 2
    return new_img, nh, nw, step_h, step_w, scale


def preprocess_image(source_image):
    """
    return batch, image_norm, nh, nw, step_h, step_w, scale
    """
    image = source_image.copy()
    if image.shape[2] == 4:
        # convert the image from RGBA2RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    w, h, c = image.shape
    image_resize = cv2.resize(image, (int(h / 2.5), int(w / 2.5)))
    image = rotate_image(image, get_angle_rotate(image_resize))

    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)  #
    image = cv2.erode(image, kernel=np.ones((2, 2), np.uint8))  #
    image, nh, nw, step_h, step_w, scale = letterbox_image(image, (1024, 1024))

    image = image / image.max()
    image_norm = image.copy()
    batch = torch.tensor(image[np.newaxis, :], dtype=torch.float).permute(0, 3, 1, 2)
    return batch, image_norm, nh, nw, step_h, step_w, scale


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return corrected


def determine_score(arr):
    histogram = np.sum(arr, axis=2, dtype=float)
    score = np.sum((histogram[..., 1:] - histogram[..., :-1]) ** 2, axis=1, dtype=float)
    return score


def get_angle_rotate(image, delta=0.2, limit=3.2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    angles = np.arange(-limit, limit + delta, delta)
    img_stack = np.stack([rotate_image(thresh, angle) for angle in angles], axis=0)
    scores = determine_score(img_stack)
    best_angle = angles[np.argmax(scores)]
    # corrected = rotate_image(image, best_angle)
    return best_angle


def sort_table_predict(predicts):
    list_pred = []
    for t_pred in predicts:
        x1, y1, x2, y2 = t_pred[:4].detach().cpu().numpy()
        list_pred.append({"x": x1, "y": y1, "pred": t_pred})
    sort_by_x = sorted(list_pred, key=lambda x: np.sqrt(x["x"] ** 2 + x["y"] ** 2))
    # print(sort_by_x)
    if len(sort_by_x) == 2:
        if sort_by_x[0]["y"] > sort_by_x[1]["y"] and sort_by_x[1]["x"] < 736 / 2:
            print("переставить")
            return [x["pred"] for x in [sort_by_x[0], sort_by_x[1]]]  # переставить!!!
        else:
            return [x["pred"] for x in sort_by_x]

    def sorted_with_y(list_items, sorted_list):
        if len(list_items) == 0:
            return sorted_list
        min_x = np.sqrt(list_items[0]["x"] ** 2 + list_items[0]["y"] ** 2)
        y_min = 10000
        min_item = list_items[0]
        for item in list_items:
            if item["y"] < y_min and np.sqrt(item["x"] ** 2 + item["y"] ** 2) < min_x:
                # if item["y"] < y_min:
                y_min = item["y"]
                min_item = item
        sorted_list.append(min_item)
        list_items.remove(min_item)
        sorted_with_y(list_items, sorted_list)

        return sorted_list

    sorted_predict = []
    sorted_with_y(sort_by_x, sorted_predict)

    return [x["pred"] for x in sorted_predict]


def list_float_values(coords):
    list_float = []
    for x in coords:
        try:
            if float(x["value"].replace(",", ".")):
                list_float.append(x)
        except:
            pass
    return list_float


def drop_last_line(_df):
    df = _df.frame.copy()
    # if all(df.iloc[-1, 1:] == df.iloc[0, 1:]):
    #     df = df[:-1]
    # else:  # разрывы в таблице
    # класстеризация на два кластера шага между ячейками
    # y_pos = np.array([x['y_c'] for x in _df.lst_X])
    # y_steps = y_pos[1:] - y_pos[:-1]

    # kmean = KMeans(2, init = [y_steps.min(), y_steps.max()])
    # kmean.fit(y_steps.reshape(-1, 1))
    # number_classter = kmean.predict(y_steps)

    # for
    # pass
    return df


def add_gaps_label(df):
    """
    Добавить метку начала и конца полигона для полигонов внутри одной таблицы
    """
    # print(df)
    df = df.reset_index(drop=True)
    df["O"] = np.nan
    has_gap = False
    add_next = False
    df_dict = []
    x_val = 1
    y_val = 1
    for k, line in df.iterrows():
        if k == 0:
            x_val = line[0]
            y_val = line[1]
            df.iloc[0, 2] = "О"
            df_dict.append({"X": x_val, "Y": y_val, "L": "О"})
            continue

        if (x_val == line[0]) and (y_val == line[1]):
            has_gap = True

        if add_next:
            df_dict.append({"X": line[0], "Y": line[1], "L": "К"})
            x_val = line[0]
            y_val = line[1]
            add_next = False
            continue

        if has_gap:
            has_gap = False
            add_next = True
            df_dict.append({"X": line[0], "Y": line[1], "L": np.nan})
        else:
            df_dict.append({"X": line[0], "Y": line[1], "L": np.nan})

    return pd.DataFrame(df_dict)


def check_coords_y(df):
    val = []
    start = ["142", "143", "144", "145", "146", "147", "148", "149", "150", "151"]
    last = str(df.Y.iloc[0])[:3]
    for i, x in enumerate(df.Y):
        if not str(x)[:3] in start:
            df.loc[i, "Y"] = float(last + str(x)[:3])


def save_table(tables, save_path, f_name):
    if len(tables) == 1:
        # final_frame = pd.concat([t.frame for t in tables], axis=0)
        final_frame = tables[0].frame
        # выкидывает последную координату
        # if all(final_frame.iloc[-1, 1:] == final_frame.iloc[0, 1:]):
        #    final_frame = final_frame[:-1]
    else:
        final_frame = pd.DataFrame()
        for table in tables:
            if len(table.frame) > 4:
                df = drop_last_line(table)
            else:
                df = table.frame
            final_frame = pd.concat((final_frame, df), ignore_index=True)
    # нет последней коорд
    final_frame = final_frame.round(2)
    final_frame = final_frame.astype(str)
    # final_frame["X"] = final_frame["X"].apply(lambda x: x.replace(".", ","))
    # final_frame["Y"] = final_frame["Y"].apply(lambda x: x.replace(".", ","))

    # check_coords_y(final_frame)
    final_frame = add_gaps_label(final_frame)
    final_frame.to_excel(
        Path(save_path) / f"{f_name}.xlsx",
        float_format="%.2f",
        index=False,
        header=False,
        na_rep="",
    )


def collect_ocr_result(result):
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
                    {
                        "value": w["value"],
                        "x_c": centr[0],
                        "y_c": centr[1],
                        "p_left": p1,
                        "p_right": p2,
                        "confidence": w["confidence"],
                    }
                )
    return coords


def correct_vertical_dim_table(image, y_low, y_high, check_discrepancy):
    """
    Уточнить вертикальные размеры таблицы
    """
    value = 1000 / (image[:, :, 0].sum(axis=1) + 1)
    quantile = np.quantile(value, 0.97)
    value[value > quantile] = quantile
    y_low += 1
    max_v = np.sort(value)[-25:].mean()
    index_start = 0
    index_last = 1
    for i, x in enumerate(value):
        # не работает
        if abs(x - max_v) / (value.max() - value.min() + 0.001) * 100 < 15:
            if index_start == 0:
                index_start = i
            index_last = i
    if index_start > y_low:
        index_start = y_low
    if index_last < y_high:
        index_last = y_high
    # если сильно большое расхождение по высоте
    if check_discrepancy:
        if index_start / y_low * 100 < 30:
            delta = (y_low - index_start) * 0.3
            if delta > abs(y_high - y_low):
                delta = abs(y_high - y_low)
            index_start = int(y_low - delta * 0.3)

        if y_high / index_last * 100 > 30:
            delta = abs((index_last - y_high)) * 0.3
            if delta > abs(y_high - y_low):
                delta = abs(y_high - y_low)
            index_last = int(y_high + delta * 0.3)
    return index_start, index_last


def process_directory(root):
    root = Path(root)
    print(os.getcwd())
    # model = YOLO('./best.pt')
    for p in Path("./").glob("*.onnx"):
        path2model = str(p)
        print(p)
    model = ort.InferenceSession(path2model)

    # ocr = ocr_predictor(
    #     det_arch="db_resnet50",
    #     reco_arch="crnn_mobilenet_v3_large",
    #     pretrained=True,
    #     detect_language=False,
    #     detect_orientation=False,
    # )
    # checkpoint = torch.load(
    #     "../doctr/db_resnet50_20240110-162426.pt", map_location="cuda"
    # )
    # ocr.det_predictor.model.load_state_dict(checkpoint)
    try:
        ocr = ocr_predictor(
            reco_arch="crnn_mobilenet_v3_large",
            pretrained=False,
            detect_language=True,
            detect_orientation=False,
        )

        checkpoint_head = torch.load(
            "./crnn_mobilenet_v3_large_pt-f5259ec2.pt", map_location="cpu"
        )
        checkpoint_dec = torch.load("./db_resnet50-ac60cadc.pt", map_location="cpu")

        ocr.det_predictor.model.load_state_dict(checkpoint_dec)
        ocr.reco_predictor.model.load_state_dict(checkpoint_head)
        print("Модель распознавания текста загружена")
    except Exception as e:
        ocr = ocr_predictor(
            reco_arch="crnn_mobilenet_v3_large",
            pretrained=False,
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
        files = list(Path(root).rglob("*"))
        files = list(
            filter(lambda x: True if x.suffix in IMAGE_EXTENTIONS else False, files)
        )
    print("Количество файлов ", len(files))
    for pdf_file in files:
        try:
            images = []

            dpi = 180  # choose desired dpi here
            zoom = dpi / 72  # zoom factor, standard: 72 dpi
            magnify = fitz.Matrix(zoom, zoom)  # magnifies in x, resp. y direction
            doc = fitz.open(pdf_file)  # open document
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=magnify)  # render page to an image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            # images = convert_from_path(
            #     pdf_file,
            #     # poppler_path=r".\poppler-23.11.0\Library\bin"
            #     poppler_path="./poppler-23.11.0/Library/bin",
            # )
        except Exception as e:
            # images = list(Path(root).rglob("*"))
            # images = list(
            #     filter(
            #         lambda x: True if x.suffix in IMAGE_EXTENTIONS else False, images
            #     )
            # )
            # print(e)
            images = [cv2.imread(pdf_file)]

        num_table = 0
        tables = []
        total_quality = False
        for image in images:
            if isinstance(image, Path):
                img = cv2.imread(str(image))
            else:
                # img = Image.frombytes("RGB", [image.width, image.height], image.samples)
                img = np.array(image)

            batch, or_img, nh, nw, step_h, step_w, scale = preprocess_image(
                img
            )  # ошибка letter box обратно в исходные координаты

            h_orig, w_orig = img.shape[:2]
            # h_crop, w_crop = or_img.shape[:2]
            # h_rate = h_orig / h_crop
            # w_rate = w_orig / w_crop
            outputs = model.run(None, {"images": batch.numpy()})  # Print Result

            # preds = model.predict(img, classes = [0]) Yolo

            predict = non_max_suppression(
                torch.tensor(outputs[0]), conf_thres=0.5, iou_thres=0.1
            )[0]
            _cls = predict[:, 5]
            table_predicts = predict[_cls == 0]

            table_predict = sort_table_predict(table_predicts)
            bad_quality = False

            last_table_df = None
            for t_pred in table_predict:
                # if len(table_predict)>2:
                #     if

                xl1, yl1, xl2, yl2 = t_pred[:4]
                x1, y1, x2, y2 = letterxyxy2org(
                    np.float64(xl1),
                    np.float64(yl1),
                    np.float64(xl2),
                    np.float64(yl2),
                    h_orig,
                    w_orig,
                    nh,
                    nw,
                    step_h,
                    step_w,
                    scale,
                )
                # print(x1, y1, x2, y2)
                down_cor = int(abs(y2 - y1) * 0.05)
                # img_crop = img[y1 - down_cor : y2 + down_cor, x1 - 10 : x2 + 10, :]

                y_low = y1 - down_cor if y1 - down_cor > 0 else 0
                y_high = y2 + down_cor if y2 + down_cor < h_orig else h_orig

                if len(table_predict) == 1:
                    index_start, index_last = correct_vertical_dim_table(
                        img[:, x1:x2], y_low, y_high, True
                    )
                elif len(table_predict) == 2:
                    crds_tables = np.array(table_predicts[:, :4])

                    if crds_tables[0][0] > crds_tables[1][0]:
                        a = crds_tables[0].copy()
                        crds_tables[0] = crds_tables[1]
                        crds_tables[1] = a
                    a1, a2 = crds_tables[0][0], crds_tables[0][2]
                    b1, b2 = crds_tables[1][0], crds_tables[1][2]
                    if (b1 - a2) > 0:
                        index_start, index_last = correct_vertical_dim_table(
                            img[:, x1:x2], y_low, y_high, True
                        )
                    else:
                        if abs(b1 - a2) / (a2 - a1) * 100 < 15:
                            index_start, index_last = correct_vertical_dim_table(
                                img[:, x1:x2], y_low, y_high, True
                            )
                        else:
                            index_start, index_last = correct_vertical_dim_table(
                                img[:, x1:x2], y_low, y_high, False
                            )
                else:
                    index_start, index_last = correct_vertical_dim_table(
                        img[:, x1:x2], y_low, y_high, False
                    )

                img_crop = img[index_start:index_last, x1:x2]
                # yolo
                # x1,y1, x2, y2 = b.xyxy.detach().cpu().numpy()[0]
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # img_crop = cv2.erode(img_crop, kernel=np.ones((2, 2), np.uint8))
                # save crop

                # cv2.imwrite(
                #     f"./results/crop/{Path(pdf_file).stem}_{np.random.randint(500)}.jpg",
                #     img_crop,
                # )
                try:
                    result = ocr([img_crop])
                    coords = collect_ocr_result(result)
                    # cчитаем уверенность сети
                    score_one = np.array(
                        [x["confidence"] for x in list_float_values(coords)]
                    ).mean()
                except Exception as e:
                    print("Неудалось", pdf_file, " ", e)
                    continue
                img_tensor = (
                    torch.tensor(img_crop / img_crop.max())
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                )
                if score_one < 0.96 or brisque(img_tensor) < 56:
                    print("Низкое качество")
                    bad_quality = True
                    total_quality = True
                else:
                    img_crop = cv2.erode(img_crop, kernel=np.ones((2, 2), np.uint8))
                    result = ocr([img_crop])
                    coords2 = collect_ocr_result(result)

                    score = np.array(
                        [x["confidence"] for x in list_float_values(coords2)]
                    ).mean()
                    if score > score_one:
                        coords = coords2

                try:
                    table = Table(ocr, img_crop)
                    cells = table.from_coords(coords)
                    print(pdf_file)
                    if isinstance(img, np.ndarray):
                        # table.frame.to_csv(
                        #     f"./results/{pdf_file.stem}_table_{num_table}.csv",
                        #     sep=";",
                        #     index=False,
                        # )
                        pass
                    else:
                        table.frame.to_csv(
                            f"./results/{image.stem}_table_{num_table}.csv",
                            sep=";",
                            index=False,
                        )
                    num_table += 1
                    if len(table.lst_X) > 0:
                        if isinstance(last_table_df, pd.DataFrame):
                            df = table.frame
                            if len(last_table_df) != len(df):  # совпадение
                                last_table_df = df
                                tables.append(table)
                            else:
                                # for x, y in zip(
                                #     last_table_df.X.to_list(), df.X.to_list()
                                # ):
                                #     print(x, y)
                                if last_table_df.X.tolist() == df.X.tolist():
                                    print("Совпадение таблиц")
                                else:
                                    last_table_df = df
                                    tables.append(table)
                        else:
                            tables.append(table)
                            last_table_df = table.frame

                except Exception as e:
                    print("Неудалось", pdf_file, " ", e)
            # after tables t_pred
        if len(tables) > 0:
            if total_quality:
                file_name = f"Плохое_качество_изображения_{pdf_file.stem}"
            else:
                file_name = pdf_file.stem

            # file_name = pdf_file.stem
            save_table(tables, "./results", file_name)
            # final_frame = pd.concat(2, axis=0)
            # final_frame.to_csv(f"./results/{pdf_file.stem}.csv", index=False)


if __name__ == "__main__":
    # process_directory("./input/")
    process_directory("./bad_example/")

# process_directory("/storage/reshetnikov/sber_table/dataset/hard/")
# process_directory("/storage/reshetnikov/sber_table/dataset/tabl/")
