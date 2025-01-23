import cv2

def change_brightness(img, value=0):
    """
    Изменяет яркость изображения.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)  # Изменение значения яркости
    v[v > 255] = 255       # Ограничение значения в пределах 0-255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def change_contrast(img, contrast=0):
    """
    Изменяет контрастность изображения.
    """

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)  # Применение взвешенного среднего для изменения контраста
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def change_size(img, scale_x=1.5, scale_y=1.5):
    """
    Изменяет размер изображения.
    """

    img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)  # Изменение размера с интерполяцией
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def change_angle_of_inclination(img, angle, scale_w=1.0, scale_h=1.0):
    """
    Поворачивает изображение на заданный угол наклона.
    """

    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Создание матрицы поворота

    img = cv2.warpAffine(img, rotation_matrix, (int(scale_w * w), int(scale_h * h)))  # Применение матрицы поворота
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def create_variants(img, func, step=1, min_dif=-100, max_dif=100):
    """
    Создаёт варианты изображений, изменённых с помощью заданной функции.
    """

    result = {}
    for i in range(min_dif, max_dif + 1, step):
        result[f"{func.__name__}_{i}"] = func(img, i)  # Применение функции с заданными параметрами

    return result


def create_size_variants(img, step=1, min_dif=1, max_dif=200):
    """
    Создаёт варианты изображений с различными масштабами.
    """

    result = {}
    for i in range(min_dif, max_dif + 1, step):
        for j in range(min_dif, max_dif + 1, step):
            scale_w = i / 100
            scale_h = j / 100

            result[f"w-{i}_h-{j}"] = change_size(img, scale_w, scale_h)  # Изменение размера с заданными коэффициентами

    return result


def apply_gaussian_blur(img, kernel_size=(5, 5)):
    """
    Применяет гауссовский размытие к изображению.
    """

    img = cv2.GaussianBlur(img, kernel_size, 0)  # Применение гауссовского размытия
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def create_blur_variants(img, step=2, min_dif=1, max_dif=21):
    """
    Создаёт варианты изображений с различными степенями размытия.
    """

    result = {}
    for i in range(min_dif, max_dif + 1, step):
        kernel_size = (i, i)
        result[f"blur_{i}"] = apply_gaussian_blur(img, kernel_size)  # Применение размытия с заданным размером ядра

    return result


def dataset(image_name):
    """
    Собирает набор вариантов изображений с различными параметрами.
    """
    img = cv2.imread(image_name)

    brightness = create_variants(img, change_brightness, step=10, min_dif=-100, max_dif=100)  # Изменение яркости
    contrast = create_variants(img, change_contrast, step=10, min_dif=-100, max_dif=100)  # Изменение контраста
    angle = create_variants(img, change_angle_of_inclination, step=5, min_dif=-45, max_dif=45)  # Изменение угла наклона
    size = create_size_variants(img, step=10, min_dif=60, max_dif=160)  # Изменение размера
    blur = create_blur_variants(img, step=2, min_dif=1, max_dif=10)  # Применение размытия

    return brightness, contrast, angle, size, blur
