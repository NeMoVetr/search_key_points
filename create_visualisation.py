
from matplotlib import pyplot as plt
import os
import numpy as np


metric_name_ru = ['Количество соответствий', 'Коэффициент удачных соответствий', 'Коэффициент найденных точек','Косинусная схожесть','Среднее расстояние сопоставления', 'Время обработки']
metric_name_code = ['num_matches','matching_ratio', 'found_points_ratio', 'cosine_similarity', 'matching_distance', 'execution_time']


def plot_histogram(folder_name, name, x, y):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlabel('Комбинация дескриптор + алгоритм соответствия')
    ax.set_ylabel(f"{name}", color='k')
    ax.bar(x, y, color='m', alpha=0.6, label=f"{name}")
    ax.tick_params(axis='y', labelcolor='k')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha='right')
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Оставить место сверху
    plt.title(f"{name}", y=1.05)
    plt.savefig(f"{folder_name}/result/Гистограммы/{name}.png")


def plot_graphic(x_labels, y_labels, matchers, metric_name, type_name, descriptor, folder_name):

    colors = ['red','orange','yellow','green','cyan','blue','purple','magenta']

    fig, ax = plt.subplots(figsize=(15, 7))
    plt.title(f"{type_name} {descriptor}")
    ax.set_xlabel("Проценты от изначальной картинки")
    ax.set_ylabel(metric_name)
    for i in range(len(x_labels)):
        ax.plot(x_labels[i], y_labels[i], color=colors[i], marker='o', label=matchers[i])
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Оставить место сверху
    plt.legend()
    plt.savefig(f'{folder_name}/result/{type_name}/{descriptor}/{type_name}_{metric_name}_{descriptor}.png')


def find_metric(metric_name, aggregated_metrics, x_labels, matchers, descriptor):
    new_metrics = []
    new_x_labels = []

    for matcher in matchers:
        metric = []
        new_label = []

        for x_label in x_labels:
            for dm in aggregated_metrics.keys():
                if dm[1][0] == descriptor and dm[1][1] == matcher and int(dm[0].split("_")[-1]) == x_label:

                    metric.append(aggregated_metrics[dm].get(metric_name, 0) )
                    
                    new_label.append(x_label)

        new_metrics.append(metric)
        new_x_labels.append(new_label)

    return new_x_labels, new_metrics


def plot_graphics_for_descriptor(folder_name, name, aggregated_metrics, x_labels, descriptor):

    if not os.path.isdir(f"{folder_name}/result/{name}/{descriptor}"):
        os.mkdir(f"{folder_name}/result/{name}/{descriptor}")

    matchers = []

    for x_label in x_labels:
        for i in [dm[1][1] for dm in aggregated_metrics.keys() if dm[1][0] == descriptor and int(dm[0].split("_")[-1]) == x_label]:
            matchers.append(i)
        
    matchers = list(set(matchers))

    for i in range(len(metric_name_code)):
        metric_labels, metric  = find_metric(metric_name_code[i], aggregated_metrics, x_labels, matchers, descriptor)
        plot_graphic(metric_labels,metric,matchers, metric_name_ru[i], name, descriptor, folder_name)


def plot_all_graphics(aggregated_metrics, name, folder_name):

    if not os.path.isdir(f"{folder_name}/result/{name}/Среднее"):
        os.mkdir(f"{folder_name}/result/{name}/Среднее")

    x_labels = sorted(set([int(dm[0].split("_")[-1]) for dm in aggregated_metrics.keys()]))

    descriptors = list(set([dm[1][0] for dm in aggregated_metrics.keys()]))

    # Список комбинаций
    detector_matcher = list(aggregated_metrics.keys())

    for descriptor in descriptors:
        plot_graphics_for_descriptor(folder_name, name, aggregated_metrics, x_labels, descriptor)

    for i in range(len(metric_name_code)):
        metric = []
        metric_labels = []
        for descriptor in descriptors:

            descriptor_metric = []
            descriptor_metric_labels = []

            for x_label in x_labels:

                new_element = [aggregated_metrics[dm].get(metric_name_code[i], 0) for dm in detector_matcher if dm[1][0] == descriptor and int(dm[0].split("_")[-1]) == x_label] 
                descriptor_metric.append(np.mean(new_element))
                descriptor_metric_labels.append(x_label)
            
            metric.append(descriptor_metric)
            metric_labels.append(descriptor_metric_labels)

        plot_graphic(metric_labels,metric,descriptors, metric_name_ru[i], name, 'Среднее', folder_name)
        

def plot_metrics(aggregated_metrics, folder_name, name=None):
    """
    Построение гистограмм и графиков
    """

    if not os.path.isdir(f"{folder_name}/result/Гистограммы"):
        os.mkdir(f"{folder_name}/result/Гистограммы")

    

    if name:
        if not os.path.isdir(f"{folder_name}/result/{name}"):
            os.mkdir(f"{folder_name}/result/{name}")
            
        plot_all_graphics(aggregated_metrics, name, folder_name)
        return 0

    # Список комбинаций
    detector_matcher = list(aggregated_metrics.keys())  

    for i in range(len(metric_name_code)):

        x_labels = [f"{dm[0]} + {dm[1]}" for dm in detector_matcher]

        metric = [aggregated_metrics[dm].get(metric_name_code[i], 0) for dm in detector_matcher]  

        plot_histogram(folder_name, metric_name_ru[i], x_labels, metric)
