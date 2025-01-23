import itertools

import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from generegate_dataset import dataset


def initialize_methods():
    """
    Инициализация дескрипторов и алгоритмов сопоставления
    """

    detectors = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(),
        'KAZE': cv2.KAZE_create(),
        'AKAZE': cv2.AKAZE_create(),
        'AGAST': cv2.AgastFeatureDetector_create(),
        'GFTT': cv2.GFTTDetector_create(),
        'MSER': cv2.MSER_create(),
        'FAST': cv2.FastFeatureDetector_create(),

    }

    bf_matchers = {
        'BF_L2': cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
        'NORM_HAMMING2': cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False),
        'NORM_L2SQR': cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=False)

    }

    # Дополнительные дескрипторы для методов без собственного дескриптора
    descriptors = {
        'ORB': cv2.ORB_create(),
        'BRISK': cv2.BRISK_create(),
    }

    flann_matchers = {
        'FLANN_LSH': cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), {}),
        'FLANN_KDTree': cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), {}),
        'FLANN_KMeans': cv2.FlannBasedMatcher(dict(algorithm=0, branching=32, iterations=7, checks=50), {})

    }

    return detectors, descriptors, bf_matchers, flann_matchers


def lowes_ratio_test(knn_matches):
    """
    Коэфициент Лоу
    """

    good_matches = 0
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue
        m1, m2 = match_pair[0], match_pair[1]
        if m1.distance < 0.8 * m2.distance:
            good_matches += 1
    return good_matches


def evaluate_methods(image1, image2, detectors, descriptors, bf_matchers, flann_matchers):
    """
    Оценка алгоритмов
    """

    metrics = {}

    # Все комбинации методов
    detector_combinations = []
    for detector_name, detector in detectors.items():
        if detector_name in ['FAST', 'GFTT','AGAST' ]:
            for desc_name, desc in descriptors.items():
                detector_combinations.append((f"{detector_name}+{desc_name}", detector, desc))
        else:
            detector_combinations.append((detector_name, detector, None))

    matcher_combinations = itertools.chain(bf_matchers.items(), flann_matchers.items())
    all_combinations = itertools.product(detector_combinations, matcher_combinations)

    # Проходимация по всем комбинациям
    for (detector_name, detector, descriptor), (matcher_name, matcher) in all_combinations:
        start_time = time.time()  # Засекаем время начала

        # Обнаружение ключевых точек
        keypoints1 = detector.detect(image1, None)
        keypoints2 = detector.detect(image2, None)

        # Если у детектора нет дескриптора, используем внешний дескриптор
        if descriptor:
            _, descriptors1 = descriptor.compute(image1, keypoints1)
            _, descriptors2 = descriptor.compute(image2, keypoints2)
        else:
            try:
                keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
                keypoints2, descriptors2 = detector.detectAndCompute(image2, None)
            except cv2.error:
                continue

        if descriptors1 is None or descriptors2 is None:
            continue


        try:
            # Сопоставление
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

            num_matches = len(matches)  # Количество соответствий

            # Количество удачных соответствий
            num_good_matches = lowes_ratio_test(matches)

            # Коэффициент удачных соответствий
            matching_ratio = num_good_matches / num_matches if num_matches > 0 and num_good_matches > 0 else 0

            num_keypoints1 = len(keypoints1)
            num_keypoints2 = len(keypoints2)

            # Коэффициент повторяемости
            repeatability = num_matches / min(num_keypoints1, num_keypoints2) if min(num_keypoints1,
                                                                                     num_keypoints2) > 0 else 0
            # Коэффициент соответствия
            matching_score = num_good_matches / num_matches if num_matches > 0 else 0

            # Инициализация метрик
            if (detector_name, matcher_name) not in metrics:
                metrics[(detector_name, matcher_name)] = {
                    'num_matches': [],
                    'repeatability': [],
                    'matching_score': [],
                    'matching_ratio': [],
                    'execution_time': [],
                }
            # Добавляем метрики
            metrics[(detector_name, matcher_name)]['num_matches'].append(num_matches)
            metrics[(detector_name, matcher_name)]['repeatability'].append(repeatability)
            metrics[(detector_name, matcher_name)]['matching_score'].append(matching_score)
            metrics[(detector_name, matcher_name)]['matching_ratio'].append(matching_ratio)
            end_time = time.time()  # Засекаем время окончания
            metrics[(detector_name, matcher_name)]['execution_time'].append(
                end_time - start_time)  # Добавляем время выполнения


        except cv2.error:
            pass

    return metrics


def aggregate_metrics(all_metrics):
    """
    Агрегация метрик
    """

    aggregated_metrics = {}
    for key, values in all_metrics.items():
        aggregated_metrics[key] = {
            'num_matches': np.mean(values.get('num_matches', [0])),
            'repeatability': np.mean(values.get('repeatability', [0])),
            'matching_score': np.mean(values.get('matching_score', [0])),
            'matching_ratio': np.mean(values.get('matching_ratio', [0])),
            'execution_time': np.mean(values.get('execution_time', [0]))
        }

    return aggregated_metrics


def plot_metrics(aggregated_metrics):
    """
    Построение гистограмм и графиков
    """

    detector_matcher = list(aggregated_metrics.keys())  # Список комбинаций

    num_matches = [aggregated_metrics[dm].get('num_matches', 0) for dm in detector_matcher]  # Количество соответствий
    matching_ratio = [aggregated_metrics[dm].get('matching_ratio', 0) for dm in
                      detector_matcher]  # Коэффициент удачных соответствий

    processing_times = [aggregated_metrics[dm].get('execution_time', 0) for dm in
                        detector_matcher]  # Скорость выполнения

    x_labels = [f"{dm[0]} + {dm[1]}" for dm in detector_matcher]

    # Гистограмма коэффициента удачных соответствий
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.set_xlabel('Комбинация дескриптор + алгоритм соответствия')
    ax1.set_ylabel('Коэффициент удачных соответствий', color='b')
    ax1.bar(x_labels, matching_ratio, color='b', alpha=0.6, label='Коэффициент удачных соответствий')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Оставить место сверху
    plt.title('Коэффициент удачных соответствий', y=1.05)
    plt.savefig('Graphics image 3/Коэффициент удачных соответствий.png')

    # График количества соответствий
    fig, ax_2 = plt.subplots(figsize=(15, 7))
    ax_2.set_xlabel('Комбинация дескриптор + алгоритм сопоставления')
    ax_2.set_ylabel('Количество соответствий', color='g')
    ax_2.plot(x_labels, num_matches, color='g', marker='o', label='Количество соответствий')
    ax_2.tick_params(axis='y', labelcolor='g')
    ax1.set_xticks(range(len(x_labels)))
    ax_2.set_xticklabels(x_labels, rotation=45, ha='right')
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Оставить место сверху
    plt.title('Количество соответствий', y=1.05)
    plt.savefig('Graphics image 3/Количество соответствий.png')

    # Гистограмма времени работы
    fig, ax5 = plt.subplots(figsize=(15, 7))
    ax5.set_xlabel('Комбинация дескриптор + алгоритм сопоставления')
    ax5.set_ylabel('Время (сек.)', color='c')
    ax5.bar(x_labels, processing_times, color='c', alpha=0.6, label='Время обработки')
    ax5.tick_params(axis='y', labelcolor='c')
    ax1.set_xticks(range(len(x_labels)))
    ax5.set_xticklabels(x_labels, rotation=45, ha='right')
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Оставить место сверху
    plt.title('Время обработки', y=1.05)
    plt.savefig('Graphics image 3/Время обработки.png')


def plot_metrics_different(aggregated_metrics, name):
    """
    Построение графиков метрик повторяемости и оценки сопоставления
    """
    x_labels = []
    repeatabilities = []
    matching_scores = []

    # Вычисления средних значений
    for i in aggregated_metrics.keys():
        if int(i[0]) not in x_labels:
            x_labels.append(int(i[0]))

    for i in x_labels:
        sum_repeatabilities = 0
        number_repeatabilities = 0
        for j in aggregated_metrics.keys():
            if int(j[0]) == i:
                number_repeatabilities += 1
                sum_repeatabilities += aggregated_metrics[j].get('repeatability', 0)

        repeatabilities.append(sum_repeatabilities / number_repeatabilities)

    for i in x_labels:
        sum_matching_scores = 0
        number_matching_scores = 0
        for j in aggregated_metrics.keys():
            if int(j[0]) == i:
                number_matching_scores += 1
                sum_matching_scores += aggregated_metrics[j].get('matching_score', 0)

        matching_scores.append(sum_matching_scores / number_matching_scores)

    fig, ax1 = plt.subplots()
    plt.title(name)
    ax1.set_xlabel("Проценты от изначальной картинки")
    ax1.set_ylabel('Повторяемость')
    ax1.plot(x_labels, repeatabilities, color='r', marker='o')
    plt.savefig(f'Graphics image 3/Повторяемость_{name}.png')

    fig, ax2 = plt.subplots()
    plt.title(name)
    ax2.set_xlabel("Проценты от изначальной картинки")
    ax2.set_ylabel('Оценка сопоставления')
    ax2.plot(x_labels, matching_scores, color='b', marker='o')
    plt.savefig(f'Graphics image 3/Оценка сопоставления_{name}.png')


def load_image(dataset_dict):
    """
    Загрузка изображений из массива словарей искажённых изображений
    """

    general_vocabulary = {}
    for img_dict in dataset_dict:
        for key, value in img_dict.items():
            general_vocabulary[key] = value
    return general_vocabulary


def main():
    """
    Главная функция
    """

    original_image = "image/3.png"  # Оригинальное изображение

    distorted_images = load_image(dataset(original_image))  # Искажённые изображения
    original_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)  # Оригинальное изображение в оттенках серого

    # Инициализация дескрипторов и алгоритмов
    detectors, descriptors, bf_matchers, flann_matchers = initialize_methods()

    all_metrics = {}
    all_metrics_brightness, all_metrics_contrast, all_metrics_blur = {}, {}, {}

    # Проход по всем изображениям
    for distorted_name, distorted_image in distorted_images.items():

        metrics = evaluate_methods(original_image, distorted_image, detectors, descriptors, bf_matchers, flann_matchers)

        # Проход по всем метрикам
        for key, values in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = {k: [] for k in values}
            for metric, value in values.items():
                all_metrics[key][metric].extend(value)

        if distorted_name.split("_")[1] == "contrast":
            num = distorted_name.split("_")[-1]
            for key, values in metrics.items():
                if key not in all_metrics_contrast:
                    all_metrics_contrast[(num, key)] = {k: [] for k in values}
                for metric, value in values.items():
                    all_metrics_contrast[(num, key)][metric].extend(value)

        if distorted_name.split("_")[1] == "brightness":
            num = distorted_name.split("_")[-1]
            for key, values in metrics.items():
                if key not in all_metrics_brightness:
                    all_metrics_brightness[(num, key)] = {k: [] for k in values}
                for metric, value in values.items():
                    all_metrics_brightness[(num, key)][metric].extend(value)

        if distorted_name.split("_")[0] == "blur":
            num = distorted_name.split("_")[-1]
            for key, values in metrics.items():
                if key not in all_metrics_blur:
                    all_metrics_blur[(num, key)] = {k: [] for k in values}
                for metric, value in values.items():
                    all_metrics_blur[(num, key)][metric].extend(value)

    # Построение графиков
    aggregated_metrics = aggregate_metrics(all_metrics)
    plot_metrics(aggregated_metrics)

    aggregated_metrics = aggregate_metrics(all_metrics_brightness)
    plot_metrics_different(aggregated_metrics, "Яркость")

    aggregated_metrics = aggregate_metrics(all_metrics_contrast)
    plot_metrics_different(aggregated_metrics, "Контраст")

    aggregated_metrics = aggregate_metrics(all_metrics_blur)
    plot_metrics_different(aggregated_metrics, "Размытие")


if __name__ == "__main__":
    main()
    print("Program: End")
