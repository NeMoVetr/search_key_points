import itertools, os, time, cv2, csv, warnings

import numpy as np

from multiprocessing import Pool, cpu_count

from sklearn.metrics.pairwise import cosine_similarity

from generegate_dataset import load_image
from create_visualisation import plot_metrics

from initializer import initialize_methods

warnings.filterwarnings("ignore")

points_columns = ["Комбинация дескриптор + алгоритм сопоставления", "Количество найденных точек"]


def save_result_image(image1, image2, key_points1, key_points2, matches, output_path, folder_name):
    """
    Сохранение искажённых изображений и количество найденных точек
    """

    with open(f"{folder_name}/result/saved_points.csv", 'a', newline='', encoding='UTF-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=points_columns)

        writer.writerow({
            points_columns[0]: output_path.split("/")[2].split(".")[0],
            points_columns[1]: len(matches)
        })

    matched_image = cv2.drawMatches(image1, key_points1, image2, key_points2, matches[:600], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite(output_path, matched_image)


def lowes_ratio_test(knn_matches):
    """
    Коэффициент Лоу
    """

    good_matches = 0

    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue

        m1, m2 = match_pair[0], match_pair[1]
        if m1.distance < 0.8 * m2.distance:
            good_matches += 1

    return good_matches


def calculate_mean_matching_distance(keypoints1, keypoints2, matches):
    """
    Вычисление среднего Евклидова расстояния между сопоставленными точками.
    """
    distances = []

    for match_pair in matches:
        if len(match_pair) < 1:
            continue  # Пропускаем пустые пары

        match = match_pair[0]  # Берём лучший матч
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        # Координаты ключевых точек
        pt1 = keypoints1[idx1].pt
        pt2 = keypoints2[idx2].pt

        # Евклидово расстояние
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
        distances.append(distance)

    # Среднее расстояние
    mean_distance = np.mean(distances) if distances else 0
    return mean_distance


def calculate_cosine_similarity(descriptors1, descriptors2, matches):
    """
    Вычисление средней косинусной схожести между дескрипторами сопоставленных точек.
    """
    cosine_similarities = []

    for match_pair in matches:
        if len(match_pair) < 1:
            continue  # Пропускаем пустые пары

        match = match_pair[0]  # Берём лучший матч
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        # Дескрипторы сопоставленных точек
        desc1 = descriptors1[idx1]
        desc2 = descriptors2[idx2]

        # Вычисление косинусной схожести
        similarity = cosine_similarity([desc1], [desc2])[0, 0]
        cosine_similarities.append(similarity)

    # Средняя косинусная схожесть
    mean_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else 0
    return mean_cosine_similarity


def evaluate_methods(image1, image2, detectors, descriptors, bf_matchers, flann_matchers, folder_name,
                     distorted_name=None) -> dict:
    """
    Оценка дескрипторов и алгоритмов сопоставления
    """

    metrics = {}

    # Все комбинации методов
    detector_combinations = []

    for detector_name, detector in detectors.items():
        if detector_name in ['FAST', 'GFTT']:
            for desc_name, desc in descriptors.items():
                detector_combinations.append((f"{detector_name}+{desc_name}", detector, desc))
        else:
            detector_combinations.append((detector_name, detector, None))

    matcher_combinations = itertools.chain(bf_matchers.items(), flann_matchers.items())
    all_combinations = itertools.product(detector_combinations, matcher_combinations)

    # Проходим по всем комбинациям
    for (detector_name, detector, descriptor), (matcher_name, matcher) in all_combinations:
        start_time = time.time()  # Засекаем время начала

        # Обнаружение ключевых точек
        key_points1 = detector.detect(image1, None)
        key_points2 = detector.detect(image2, None)

        # Если у детектора нет дескриптора, используем внешний дескриптор
        if descriptor:
            _, descriptors1 = descriptor.compute(image1, key_points1)
            _, descriptors2 = descriptor.compute(image2, key_points2)
        else:
            try:
                key_points1, descriptors1 = detector.detectAndCompute(image1, None)
                key_points2, descriptors2 = detector.detectAndCompute(image2, None)

            except cv2.error:
                continue

        if descriptors1 is None or descriptors2 is None:
            continue

        end_time = time.time()  # Засекаем время окончания

        try:
            # Сопоставление
            if distorted_name is not None:
                result_image_path = os.path.join(f"{folder_name}/new_images/",
                                                 f"{distorted_name}_{detector_name}_{matcher_name}.png")
            else:
                result_image_path = os.path.join(f"{folder_name}/new_images/", f"{detector_name}_{matcher_name}.png")

            matches = matcher.match(descriptors1, descriptors2)

            save_result_image(image1, image2, key_points1, key_points2, matches, result_image_path, folder_name)

            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

            # Количество соответствий
            num_matches = len(matches)

            # Количество удачных соответствий
            num_good_matches = lowes_ratio_test(matches)

            # Коэффициент удачных соответствий
            matching_ratio = num_good_matches / num_matches if num_matches > 0 and num_good_matches > 0 else 0

            # Коэффициент найденных точек
            num_key_points1 = len(key_points1)
            num_key_points2 = len(key_points2)
            found_points_ratio = num_key_points1 / num_key_points2 if num_key_points1 / num_key_points2 > 0 else 0

            # Среднее расстояние сопоставления
            matching_distance = calculate_mean_matching_distance(key_points1, key_points2, matches)

            # Средняя косинусная схожесть
            cosine_similarity = calculate_cosine_similarity(descriptors1, descriptors2, matches)

            # Инициализация метрик
            if (detector_name, matcher_name) not in metrics:
                metrics[(detector_name, matcher_name)] = {
                    'num_matches': [],
                    'found_points_ratio': [],
                    'matching_ratio': [],
                    'execution_time': [],
                    'cosine_similarity': [],
                    'matching_distance': []
                }

            # Добавляем метрики
            metrics[(detector_name, matcher_name)]['num_matches'].append(num_matches)
            metrics[(detector_name, matcher_name)]['found_points_ratio'].append(found_points_ratio)
            metrics[(detector_name, matcher_name)]['matching_ratio'].append(matching_ratio)
            metrics[(detector_name, matcher_name)]['execution_time'].append(end_time - start_time)
            metrics[(detector_name, matcher_name)]['cosine_similarity'].append(cosine_similarity)
            metrics[(detector_name, matcher_name)]['matching_distance'].append(matching_distance)


        except cv2.error:
            continue

    return metrics


def aggregate_metrics(all_metrics):
    """
    Агрегация метрик
    """

    aggregated_metrics = {}
    for key, values in all_metrics.items():
        aggregated_metrics[key] = {
            'num_matches': np.mean(values.get('num_matches', [0])),
            'found_points_ratio': np.mean(values.get('found_points_ratio', [0])),
            'matching_ratio': np.mean(values.get('matching_ratio', [0])),
            'execution_time': np.mean(values.get('execution_time', [0])),
            'cosine_similarity': np.mean(values.get('cosine_similarity', [0])),
            'matching_distance': np.mean(values.get('matching_distance', [0]))
        }

    return aggregated_metrics


def dict_metrics(metrics, distorted_name):
    dict_all_metrics = {}
    num = distorted_name.split("_")[-1]
    for key, values in metrics.items():
        if key not in dict_all_metrics:
            dict_all_metrics[(num, key)] = {k: [] for k in values}
        for metric, value in values.items():
            dict_all_metrics[(num, key)][metric].extend(value)

    return dict_all_metrics


def save_data_helper(aggregated_metrics, writer):
    detector_matcher = list(aggregated_metrics.keys())  # Список комбинаций

    num_matches = [aggregated_metrics[dm].get('num_matches', 0) for dm in detector_matcher]  # Количество соответствий
    matching_ratio = [aggregated_metrics[dm].get('matching_ratio', 0) for dm in
                      detector_matcher]  # Коэффициент удачных соответствий
    processing_times = [aggregated_metrics[dm].get('execution_time', 0) for dm in
                        detector_matcher]  # Скорость выполнения
    found_points_ratio = [aggregated_metrics[dm].get('found_points_ratio', 0) for dm in
                          detector_matcher]  # Коэффициент найденных точек
    cosine_similarity = [aggregated_metrics[dm].get('cosine_similarity', 0) for dm in
                         detector_matcher]  # Косинусная схожесть
    matching_distance = [aggregated_metrics[dm].get('matching_distance', 0) for dm in detector_matcher]

    for idx in range(len(detector_matcher)):
        # выбор строчек из файлов

        writer.writerow({
            'Комбинация дескриптор + алгоритм сопоставления': detector_matcher[idx],
            'Количество соответствий': round(num_matches[idx], 3),
            'Коэффициент удачных соответствий': round(matching_ratio[idx], 3),
            'Коэффициент найденных точек': round(found_points_ratio[idx], 3),
            'Время обработки': round(processing_times[idx], 3),
            'Среднее расстояние сопоставления': round(matching_distance[idx], 3),
            'Косинусная схожесть': round(cosine_similarity[idx], 3)
        })


def compare_distorted(folder_name, file_name_1):
    """
    Главная функция сравнения изображений
    """

    original_image = cv2.imread(f"{folder_name}/{file_name_1}",
                                cv2.IMREAD_GRAYSCALE)  # Оригинальное изображение в оттенках серого

    distorted_images = load_image(folder_name, file_name_1)  # Искажённое изображения

    # Инициализация дескрипторов и алгоритмов
    detectors, descriptors, bf_matchers, flann_matchers = initialize_methods()

    all_metrics, all_metrics_brightness, all_metrics_contrast, all_metrics_blur = {}, {}, {}, {}

    # Проход по всем изображениям
    for distorted_name, distorted_image in distorted_images.items():

        metrics = evaluate_methods(original_image, distorted_image, detectors, descriptors, bf_matchers, flann_matchers,
                                   folder_name,
                                   distorted_name=distorted_name)

        # Проход по всем метрикам
        for key, values in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = {k: [] for k in values}
            for metric, value in values.items():
                all_metrics[key][metric].extend(value)

        if distorted_name.split("_")[1] == "contrast":
            all_metrics_brightness.update(dict_metrics(metrics, distorted_name))

        elif distorted_name.split("_")[1] == "brightness":
            all_metrics_contrast.update(dict_metrics(metrics, distorted_name))

        elif distorted_name.split("_")[1] == "blur":
            all_metrics_blur.update(dict_metrics(metrics, distorted_name))

    # Построение графиков
    aggregated_metrics = aggregate_metrics(all_metrics)
    plot_metrics(aggregated_metrics, folder_name)

    aggregated_metrics_brightness = aggregate_metrics(all_metrics_brightness)
    plot_metrics(aggregated_metrics_brightness, folder_name, "Яркость")

    aggregated_metrics_contrast = aggregate_metrics(all_metrics_contrast)
    plot_metrics(aggregated_metrics_contrast, folder_name, "Контраст")

    aggregated_metrics_blur = aggregate_metrics(all_metrics_blur)
    plot_metrics(aggregated_metrics_blur, folder_name, "Размытие")

    columns = ["Комбинация дескриптор + алгоритм сопоставления", 'Количество соответствий',
               'Коэффициент удачных соответствий', 'Коэффициент найденных точек', 'Время обработки',
               'Среднее расстояние сопоставления',
               'Косинусная схожесть']

    with open(f"{folder_name}/result/saved_data.csv", 'w', newline='', encoding='UTF-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter=';')
        writer.writeheader()
        save_data_helper(aggregated_metrics, writer)

        # save_data_helper(aggregated_metrics_contrast, writer)

        # save_data_helper(aggregated_metrics_blur, writer)

        # save_data_helper(aggregated_metrics_brightness, writer)


def compare_two(folder_name, file_name_1, file_name_2):
    """
    Главная функция для обработки искажённых изображений
    """

    original_image = cv2.imread(f"{folder_name}/{file_name_1}",
                                cv2.IMREAD_GRAYSCALE)  # Оригинальное изображение в оттенках серого

    compared_image = cv2.imread(f"{folder_name}/{file_name_2}", cv2.IMREAD_GRAYSCALE)

    # Инициализация дескрипторов и алгоритмов
    detectors, descriptors, bf_matchers, flann_matchers = initialize_methods()

    all_metrics = {}

    metrics = evaluate_methods(original_image, compared_image, detectors, descriptors, bf_matchers, flann_matchers,
                               folder_name)

    # Проход по всем метрикам
    for key, values in metrics.items():
        if key not in all_metrics:
            all_metrics[key] = {k: [] for k in values}
        for metric, value in values.items():
            all_metrics[key][metric].extend(value)

    # Построение графиков
    aggregated_metrics = aggregate_metrics(all_metrics)
    plot_metrics(aggregated_metrics, folder_name)

    columns = ["Комбинация дескриптор + алгоритм сопоставления", 'Количество соответствий',
               'Коэффициент удачных соответствий', 'Коэффициент найденных точек', 'Время обработки',
               'Среднее расстояние сопоставления',
               'Косинусная схожесть']

    with open(f"{folder_name}/result/saved_data.csv", 'w', newline='', encoding='UTF-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, delimiter=';')
        writer.writeheader()
        save_data_helper(aggregated_metrics, writer)


def process_folder(args):
    """
    Параллельное обработка папок
    """

    folder_name, filename_1, filename_2 = args

    if not os.path.isdir(f"{folder_name}/result"):
        os.mkdir(f"{folder_name}/result")

    if not os.path.isdir(f"{folder_name}/new_images"):
        os.mkdir(f"{folder_name}/new_images")

    with open(f"{folder_name}/result/saved_points.csv", 'w', newline='', encoding='UTF-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=points_columns, delimiter=';')
        writer.writeheader()

    if os.path.isfile(f"{folder_name}/{filename_2}"):
        compare_two(folder_name, filename_1, filename_2)
    else:
        compare_distorted(folder_name, filename_1)


if __name__ == "__main__":
    print("Program: Start")

    filename_1 = "original_image.jpg"
    filename_2 = "compared_image.jpg"

    folder_name = "image_"

    # Подготовка списка папок
    folders = [(f"{folder_name}{i}", filename_1, filename_2) for i in range(1, 8)]

    # Использования многопроцессорности
    with Pool(cpu_count()) as pool:
        pool.map(process_folder, folders)

    print("Program: End")
