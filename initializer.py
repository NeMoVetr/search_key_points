import cv2


def initialize_methods():
    """
    Инициализация дескрипторов и алгоритмов сопоставления
    """

    detectors = {
        'ORB': cv2.ORB_create(),
        'FAST': cv2.FastFeatureDetector_create(),
        'KAZE': cv2.KAZE_create(),
        'AKAZE': cv2.AKAZE_create(),
        'AGAST': cv2.AgastFeatureDetector_create(),
        'GFTT': cv2.GFTTDetector_create(),
        'MSER': cv2.MSER_create(),
        'SIFT': cv2.SIFT_create(),
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

