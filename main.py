from ultralytics import YOLO
import cv2
from cv2 import aruco
import numpy as np

# Загружаем модель
model = YOLO("best.pt")

# Загружаем изображение
img = cv2.imread('IMG20240603175604.jpg')

# Загружаем Aruco Detector, определим параметры и загрузим предопределенный маркер (словарь)
param_markers = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# Находим маркер на изображении
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_img, aruco_dict, parameters=param_markers)

# Обведем маркер на изображении
if marker_corners:
    for ids, corners in zip(marker_IDs, marker_corners):
        int_corners = np.int0(marker_corners)
        cv2.polylines(img, int_corners, True, (0, 255, 255), 4, cv2.LINE_AA)

# Найдем площадь и периметр маркера
aruco_area = cv2.contourArea(marker_corners[0], True)
aruco_perimeter = cv2.arcLength(marker_corners[0], True)

# Отношение пикселей к сантиметрам, а также к сантиметрам квадратным
pixel_cm_ratio = aruco_perimeter / 20
pixel_cm2_ratio = aruco_area / 25

# Используем ранее загруженную модель для обнаружения выбоин
h, w, _ = img.shape
results = model.predict(img)

for r in results:
    boxes = r.boxes
    masks = r.masks

# Обработаем результаты обнаружения
if masks is not None:
    masks = masks.data.cpu()
    for seg, box in zip(masks.data.cpu().numpy(), boxes.data.cpu().numpy()):
        seg = cv2.resize(seg, (w, h))
        # Определим
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Получим координаты прямоугольника, в который вписан дефект
            x, y, x1, y1 = cv2.boundingRect(contour)
            # Обведем контуры дефекта и прямоугольника
            cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
            cv2.rectangle(img, (x, y), (x1 + x, y1 + y), (255, 0, 0), 2)
            # Рассчитаем линейные размеры и площадь дефектов
            pothole_area = 'Damaged area: ' + str(round(cv2.contourArea(contour) / pixel_cm2_ratio, 1)) + ' cm2'
            rect_width = round(x1 / pixel_cm_ratio, 1)
            rect_height = round(y1 / pixel_cm_ratio, 1)
            rect_area = 'Rectangle area: ' + str(round(rect_height * rect_width, 1)) + ' cm2'
            str_width = 'Width: ' + str(round(x1 / pixel_cm_ratio, 1)) + ' cm'
            str_height = 'Height: ' + str(round(y1 / pixel_cm_ratio, 1)) + ' cm'
            # Отобразим полученные результаты на исходное изображение
            cv2.putText(img, pothole_area, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, rect_area, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, str_width, (int(x + 5), int(y + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, str_height, (int(x + 5), int(y + 70)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# Выведем изображение на экран и запишем его в файл
cv2.imshow("Image", img)
cv2.imwrite('result.jpg', img)
cv2.waitKey(0)
