import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

from HyperTools import predict, LoadingData

# 使用 SimHei（黑体）字体，适用于中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def hex_to_rgb(hex_str):
    """将十六进制颜色字符串转换为RGB数组（0-1范围）"""
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def get_class_detail(flag):
    """
    返回 flag 数据集的string真实标签, 即(0, 1, 2, 3 ....) 所代表的真实string类别
    """
    Map = {
        'tea': ['Massonpine', 'Bambooforeste', 'Tea plante', 'Reede', 'Rice-paddye', 'Sweet potato',
                'Carawaye', 'Weedw', 'Waterbodye', 'Building/roade'],

        'xuzhou': ['Bareland-1', 'Trees', 'Bareland-2', 'Coals', 'Crops-1', 'Cement', 'Crops-2', 'Lakes', 'Red-tiles'],
        'botswana': ['Water', 'Hippo grass', 'Floodplain grasses 1', 'Floodplain grasses 2', 'Reeds', 'Riparian',
                     'Firescar', 'Island interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
                     'Short mopane', 'Mixed mopane', 'Chalcedony'],

        'loukia': ['Dense urban fabric', 'Mineral extraction site', 'Non-irrigated arable land', 'Fruit trees',
                   'Olive groves', 'Broad-leaved forest', 'Coniferous forest', 'Mixed forest',
                   'Dense sclerophyllous vegetation', 'Sparce sclerophyllous vegetation', 'Sparsely vegetated areas',
                   'Rocks and sand', 'Water', 'Coastal water'],

        'houston': ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential',
                    'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                    'Running Track'],

        'houston2018': ['Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees', 'Deciduous trees',
                        'Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings', 'Roads',
                        'Sidewalks', 'Crosswalks', 'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots',
                        'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats'],

        'indian': ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
                   'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                   'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'],

        'sali': [''
                 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth',
                 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                 'Vinyard_untrained', 'Vinyard_vertical_trellis'],

        'paviau': ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                   'Self-Blocking Bricks', 'Shadows'],

        'paviac': ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen', 'Tiles', 'Shadows', 'Meadows',
                   'Bare Soil'],

        'longkou': ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean', 'Narrow-leaf soybean', 'Rice',
                    'Water', 'Roads and houses', 'Mixed weed'],

        'hanchuan': ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum', 'Water spinach', 'Watermelon', 'Greens', 'Trees',
                     'Grass', 'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road', 'Bright object', 'Water'],

        'honghu': ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage', 'Pakchoi',
                   'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis',
                   'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
                   'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree'],

        'ksc': ['Scrub', 'Willow swamp', 'CP hammock', 'Slash pine', 'Oak/Broadleaf', 'Hardwood', 'Swamp',
                'Graminoid marsh', 'Spartina marsh', 'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water'],

        'SZUR1': ['Ficus concinna', 'Ficus macrophylla', 'Litchi chinensis', 'Dimocarpus longan',
                  'Araucaria cunninghamii', 'Acacia auriculiformis', 'Camphora officinarum', 'Ficus elastica',
                  'Livistona chinensis', 'Leucaena leucocephala', 'Roystonea regia',
                  'Mangifera indica', 'Terminalia arjuna', 'Delonix regia', 'Kigelia africana',
                  'Archontophoenix alexandrae', 'Bombax ceiba'],

        'SZUR2': ['Ficus concinna', 'Ficus macrophylla', 'Litchi chinensis', 'Araucaria cunninghamii',
                  'Acacia auriculiformis', 'Ficus elastica', 'Livistona chinensis', 'Leucaena leucocephala',
                  'Roystonea regia', 'Mangifera indica', 'Terminalia arjuna', 'Delonix regia',
                  'Kigelia africana', 'Ficus virens', 'Archontophoenix alexandrae', 'Swietenia mahagoni',
                  'Plumeria', 'Bauhinia purpurea', 'Dracontomelon duperreanum', 'Melaleuca',
                  'Casuarina equisetifolia'],
        'HC': ['Analcime', 'Plagioclase', 'Prehnite', 'High-Ca Pyroxene', 'Serpentine', 'Margarite'],
        'UP': ['Analcime', 'Bassanite', 'High-Ca Pyroxene', 'Illite/Muscovite', 'Low-Ca Pyroxene', 'Mg-Smectite',
               'Monohydrated sulfate', 'Plagioclase', 'Prehnite'],
        'NF': ['Fe-Olivine', 'Epidote', 'Chlorite', 'Bassanite', 'Illite/Muscovite', 'Mg-Carbonate', 'Plagioclase',
               'Prehnite', 'Serpentine'],
        'chi': ['Water', 'Bare soil (school)', 'Bare soil (park)', 'Bare soil (farmland)', 'Natural plants',
                'Weeds in farmland', 'Forest', 'Grass', 'Rice field (grown)', 'Rice field (first stage)', 'Row crops',
                'Plastic house', 'Manmade (non-dark)', 'Manmade (dark)', 'Manmade (blue)', 'Manmade (red)',
                'Manmade grass', 'Asphalt', 'Paved ground'],
        'dioni': ['Dense Urban Fabric', 'Mineral Extraction Sites', 'Non Irigated Arable Land', 'Fruit Trees',
                  'Olve Groves', 'Broad -leaved Forest', 'Coniferous Forest', 'Mixed Forest',
                  'Dense Sderophylous Vegetaton', 'Sparce Sderophylous Vegetation', 'Sparcely Vegetated Areas',
                  'Rocks and Sand', 'Water', 'Coastal Water'],
    }
    return Map[flag]


def get_map(model, device, all_data_loader, y, flag, path):
    """
    生成分类地图和 Ground Truth 的图像文件。

    参数:
        net (torch.nn.Module): 训练好的模型。
        device (torch.device): 模型和数据所在的设备（CPU 或 GPU）。
        all_data_loader (torch.utils.data.DataLoader): 测试数据加载器。
        y (numpy.ndarray): 实际标签，二维数组。
    """
    y_pred, y_new = predict(all_data_loader, model, device)  # 使用模型预测
    pre_labels = get_classification_map(y_pred, y)  # 生成分类地图
    x = np.ravel(pre_labels)  # 将分类地图展平
    gt = y.flatten()  # 将实际标签展平

    # 要什么颜色

    y_list = list_to_colormap(x, get_colo_list(flag))
    y_gt = list_to_colormap(gt, get_colo_list(flag))

    # 重塑为原始地图的形状
    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))

    # 保存分类地图和 Ground Truth 地图
    classification_map(y_re, y, 300, path + f'/{flag}_predictions.eps')
    classification_map(y_re, y, 300, path + f'/{flag}_predictions.png')
    classification_map(gt_re, y, 300, path + f'/{flag}_gt.png')

    print('------Get classification maps successful-------')


def AA_andEachClassAccuracy(ConfusionMatrix):
    """ return acc """
    list_diag = np.diag(ConfusionMatrix)
    list_raw_sum = np.sum(ConfusionMatrix, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        each_acc = np.nan_to_num(np.true_divide(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test, flag):
    """ Reports on the various evaluations of the trained models """
    target_names = get_class_detail(flag)

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names, zero_division=0)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def list_to_colormap(x_list, col_list):
    y = np.zeros((x_list.shape[0], 3))

    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        else:
            y[index] = col_list[int(item - 1)]
    return y


def get_colo_list(flag):
    palette = None
    if flag == 'paviau' or flag == 'xuzhou':  # PU
        row = 610
        col = 340
        palette = np.array([[216, 191, 216],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])
    elif flag == 'indian':  # IP
        row = 145
        col = 145

        palette = np.array([
            [255, 0, 0],      # Alfalfa
            [0, 255, 0],      # Corn-notill
            [0, 0, 255],      # Corn-mintill
            [255, 255, 0],    # Corn
            [0, 255, 255],    # Grass-pasture
            [255, 0, 255],    # Grass-trees
            [176, 48, 96],    # Grass-pasture-mowed
            [46, 139, 87],    # Hay-windrowed
            [160, 32, 240],   # Oats
            [255, 127, 80],   # Soybean-notill
            [127, 255, 212],  # Soybean-mintill
            [218, 112, 214],  # Soybean-clean
            [160, 82, 45],    # Wheat
            [132, 231, 161],  # Woods
            [216, 191, 216],  # Buildings-Grass-Trees-Drives
            [210, 145, 145],  # Stone-Steel-Towers
        ])
    elif flag == 'botswana':  # Botswana
        row = 1476
        col = 256
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0]])
    elif flag == 'loukia':  # Loukia  249 x 945 x 176
        row = 249
        col = 945
        palette = np.array([[237, 48, 35],  # 城市建筑 - 深红色
                            [157, 158, 159],  # 矿物提取区 - 灰色
                            [175, 175, 100],  # 非灌溉耕地 - 淡黄褐色
                            [0, 154, 61],  # 果树 - 绿色
                            [76, 114, 29],  # 橄榄林 - 橄榄绿
                            [62, 128, 0],  # 阔叶林 - 深绿色
                            [0, 97, 36],  # 针叶林 - 暗绿色
                            [52, 102, 25],  # 混交林 - 森林绿
                            [135, 184, 82],  # 密集硬叶植被 - 亮绿色
                            [172, 209, 79],  # 稀疏硬叶植被 - 浅绿色
                            [221, 225, 162],  # 稀疏植被区 - 淡黄绿色
                            [240, 196, 110],  # 岩石和沙子 - 沙色
                            [0, 181, 226],  # 水域 - 蓝色
                            [0, 120, 175]])  # 沿海水域 - 深蓝色
    elif flag == 'sali':  # Salinas
        row = 512
        col = 217
        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78],
                            [238, 234, 63],
                            [246, 187, 31],
                            [244, 127, 33],
                            [239, 71, 34],
                            [238, 33, 35],
                            [180, 31, 35],
                            [123, 18, 20]])
    elif flag == 'paviac':  # Pavia Centre
        row = 1096
        col = 715
        palette = np.array([[37, 97, 163],
                            [44, 153, 60],
                            [122, 182, 41],
                            [219, 36, 22],
                            [227, 156, 47],
                            [227, 221, 223],
                            [108, 35, 127],
                            [130, 67, 142],
                            [229, 225, 74]])
    elif flag == 'ksc':  # KSC
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
    elif flag == 'houston':  # Houston
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
    elif flag == 'hanchuan':  # Hanchuan
        row = 1217
        col = 303
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
    elif flag == 'honghu':  # Honghu
        row = 1217
        col = 303
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [0, 139, 0],
                            [37, 58, 150],
                            [47, 78, 161],
                            [123, 18, 20]])
    elif flag == 'longkou':  # Longkou
        row = 550
        col = 400
        palette = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [176, 48, 96],
            [46, 139, 87],
            [160, 32, 240]]
        )
    elif flag == 'houston2018':
        row = 601
        col = 2384
        palette = np.array([
            [0, 255, 0],
            [129, 255, 0],
            [47, 140, 88],
            [0, 139, 0],
            [0, 70, 0],
            [160, 82, 45],
            [0, 255, 255],
            [255, 255, 255],
            [216, 192, 216],
            [255, 0, 0],
            [170, 160, 150],
            [128, 128, 128],
            [160, 0, 0],
            [81, 1, 3],
            [233, 162, 25],
            [255, 255, 0],
            [238, 154, 0],
            [255, 0, 255],
            [0, 0, 255],
            [176, 196, 222]]
        )
    elif flag == 'SZUR1':
        row = 2405
        col = 3085
        palette = np.array([
            [55, 66, 61],
            [163, 132, 163],
            [249, 191, 0],
            [196, 45, 198],
            [132, 109, 211],
            [127, 255, 255],
            [119, 193, 206],
            [239, 226, 216],
            [216, 130, 221],
            [193, 124, 147],
            [211, 68, 112],
            [66, 114, 196],
            [66, 38, 119],
            [51, 175, 71],
            [183, 206, 193],
            [216, 214, 219],
            [0, 255, 0]]
        )
    elif flag == 'dioni':
        row = 0
        col = 0
        palette = np.array([
            [196, 45, 198],
            [132, 109, 211],
            [127, 255, 255],
            [119, 193, 206],
            [239, 226, 216],
            [216, 130, 221],
            [193, 124, 147],
            [211, 68, 112],
            [66, 114, 196],
            [66, 38, 119],
            [51, 175, 71],
            [183, 206, 193],
            [216, 214, 219],
            [0, 255, 0]]
        )
    elif flag == 'SZUR2':
        row = 2444
        col = 4040
        palette = np.array([
            [55, 66, 61],
            [163, 132, 163],
            [249, 191, 0],
            [132, 109, 211],
            [127, 255, 255],
            [239, 226, 216],
            [216, 130, 221],
            [193, 124, 147],
            [211, 68, 112],
            [66, 114, 196],
            [66, 38, 119],
            [51, 175, 71],
            [183, 206, 193],
            [17, 89, 102],
            [216, 214, 219],
            [255, 255, 0],
            [193, 214, 165],
            [255, 0, 255],
            [221, 216, 107],
            [153, 61, 45],
            [192, 189, 50]]
        )
    elif flag == 'HC':
        """ consists of 418×595 pixels and retains 440 bands 6 classes"""
        palette = np.array([
            [81, 1, 3],
            [233, 162, 25],
            [255, 255, 0],
            [249, 191, 0],
            [132, 109, 211],
            [127, 255, 255],
        ])
    elif flag == 'NF':
        """ dimensions of 478×593 pixels and includes 425 bands """
        palette = np.array([
            [216, 130, 221],
            [193, 124, 147],
            [211, 68, 112],
            [66, 114, 196],
            [66, 38, 119],
            [51, 175, 71],
            [183, 206, 193],
            [216, 214, 219],
            [0, 255, 0]]
        )
    elif flag == 'UP':
        """ Tianwen-1 rover 'Zhurong', has a dataset size of 478×595×432 """
        palette = np.array([
            [160, 82, 45],
            [127, 255, 0],
            [216, 191, 216],
            [238, 0, 0],
            [238, 154, 0],
            [85, 26, 139],
            [0, 139, 0],
            [37, 58, 150],
            [47, 78, 161],
        ])
    elif flag == 'tea':
        """ 茶叶数据集 """
        palette = np.array([
            [36, 100, 36],  # 马尾松 - 深绿色
            [76, 153, 0],  # 竹林 - 竹绿色
            [0, 128, 0],  # 茶树 - 绿色
            [128, 128, 0],  # 芦苇 - 橄榄色
            [153, 153, 0],  # 水稻 - 黄绿色
            [204, 153, 0],  # 红薯 - 棕黄色
            [153, 102, 51],  # 藏红花 - 棕色
            [51, 153, 51],  # 杂草 - 草绿色
            [0, 102, 204],  # 水体 - 蓝色
            [128, 128, 128],  # 建筑/道路 - 灰色
        ])
    palette = palette / 255.
    return palette


def get_classification_map(y_pred, y):
    height, width = y.shape
    cls_labels = np.zeros((height, width))

    y_pred = np.asarray(y_pred)
    total_pixels = height * width
    labelled_mask = y > 0
    labelled_pixels = int(np.count_nonzero(labelled_mask))

    if len(y_pred) == total_pixels:
        cls_labels = y_pred.reshape(height, width) + 1
    elif len(y_pred) == labelled_pixels:
        cls_labels[labelled_mask] = y_pred + 1
    else:
        raise ValueError(
            f"Prediction length {len(y_pred)} does not match full image pixels "
            f"({total_pixels}) or labelled pixels ({labelled_pixels})."
        )

    return cls_labels


def classification_map(Map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(Map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def data_info(train_label=None, val_label=None, test_label=None, start=1):
    total_train_pixel, total_val_pixel, total_test_pixel = 0, 0, 0
    """
    输出每个类别在训练集、验证集和测试集中的像素数量，并统计总数。

    参数:
    train_label: numpy.ndarray 或 None, 训练标签。
    val_label: numpy.ndarray 或 None, 验证标签。
    test_label: numpy.ndarray 或 None, 测试标签。
    start: int, 起始类别编号，默认为 1。
    """
    class_num = np.max(train_label).astype('int32')

    def print_and_sum(label_counter, label_name):
        total_pixels = 0
        for i in range(start, class_num + 1):
            count = label_counter.get(i, 0)
            print(f"class {i} in {label_name}: {count}")
            total_pixels += count
        print(f"Total pixels in {label_name}: {total_pixels}\n")
        return total_pixels

    if train_label is not None:
        train_counter = Counter(train_label.flatten())
        print("Train Label Information:")
        total_train_pixel = print_and_sum(train_counter, "train")

    if val_label is not None:
        val_counter = Counter(val_label.flatten())
        print("Validation Label Information:")
        total_val_pixel = print_and_sum(val_counter, "validation")

    if test_label is not None:
        test_counter = Counter(test_label.flatten())
        print("Test Label Information:")
        total_test_pixel = print_and_sum(test_counter, "test")

    # 总结部分
    if train_label is not None and val_label is not None and test_label is not None:
        print(f"Overall Total: Train={total_train_pixel}, Validation={total_val_pixel}, Test={total_test_pixel}")
    elif train_label is not None and val_label is not None:
        print(f"Overall Total: Train={total_train_pixel}, Validation={total_val_pixel}")
    elif train_label is not None:
        print(f"Overall Total: Train={total_train_pixel}")


def plot_spectra_by_class(hypercube, label_map, flag='indian', output_dir="Line_chart", dpi=300):
    path = os.path.join(output_dir, flag)
    if not os.path.exists(path):
        os.makedirs(path)

    """
    按类别绘制每个像素的光谱曲线，并保存为 PNG 文件。
    # 示例用法
    # 假设 hypercube 是高光谱数据，label_map 是分类标签
    # plot_spectra_by_class(hypercube, label_map)
    参数:
    - hypercube: 高光谱数据立方体 (H x W x D) 的 NumPy 数组。
    - label_map: 分类标签图 (H x W)，每个值表示一个类别。
    - output_dir: 输出文件夹，用于存储 PNG 文件，默认值为 "output"。
    - dpi: 保存图片的分辨率，默认值为 300。
    """
    # 检查输入数据
    num_to_class = get_class_detail(flag)
    if hypercube.ndim != 3 or label_map.ndim != 2:
        raise ValueError("hypercube 必须是三维数组 (H x W x D)，label_map 必须是二维数组 (H x W)。")
    if hypercube.shape[:2] != label_map.shape:
        raise ValueError("hypercube 的空间维度 (H, W) 必须与 label_map 匹配。")

    # 获取类别列表
    classes = list(set(np.unique(label_map)))
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 遍历每个类别
    # print(len(classes))
    for cls in classes:
        # 像素为未定义时候continue
        if cls == 0:
            continue
        # 获取该类别的像素坐标
        coords = np.argwhere(label_map == cls)
        if coords.size == 0:
            continue

        # 绘图
        plt.figure(figsize=(12, 8))
        for (x, y) in coords:
            spectrum = hypercube[x, y, :]
            plt.plot(range(hypercube.shape[2]), spectrum, alpha=0.5, lw=0.8)

        # 设置图像属性
        # plt.title(f"Spectral 种类:  {num_to_class[cls - 1]}  数量为: {len(coords)}", fontsize=14)
        plt.xlabel("Spectral Band Index", fontsize=12)
        plt.ylabel("Reflectance/Intensity", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        # 保存图像
        output_path = os.path.join(output_dir, flag, f"class_{cls}_{num_to_class[cls - 1]}.png")
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"类别 {cls}_{num_to_class[cls - 1]} 的光谱曲线图已保存到 {output_path}")


def plot_pixel_and_neighbors(hypercube, data_gt, x, y, output_dir="neighbor_spectra",
                             dpi=300, flag='indian'):
    """
    绘制指定像素及其周围邻域光谱，按真实类别着色

    参数:
    - hypercube: 高光谱数据立方体 (H x W x D)
    - data_gt: 真实标签的二维数组 (H x W)
    - x: 目标像素x坐标
    - y: 目标像素y坐标
    - output_dir: 输出目录
    - dpi: 图像分辨率
    - flag: 数据集标识
    """
    # 类别颜色映射
    class_colors = plt.cm.tab20(np.linspace(0, 1, 16))  # 使用tab20色板
    class_names = {
        'indian': ['未定义', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                   'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
                   'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat',
                   'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']
    }

    # 检查坐标有效性
    if not (0 <= x < hypercube.shape[0] and 0 <= y < hypercube.shape[1]):
        raise ValueError(f"坐标 ({x}, {y}) 超出数据范围。")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化画布
    plt.figure(figsize=(14, 8))

    # 记录已绘制类别
    legend_handles = {}

    # 绘制周围邻域
    neighbor_count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if (dx == 0 and dy == 0) or not (0 <= nx < hypercube.shape[0] and 0 <= ny < hypercube.shape[1]):
                continue

            # 获取类别信息
            class_id = data_gt[nx, ny]
            if class_id == 0:  # 跳过未定义类别
                continue

            # 获取颜色和类别名
            color = class_colors[class_id]
            class_name = class_names[flag][class_id]

            # 绘制光谱
            spectrum = hypercube[nx, ny, :]
            line = plt.plot(spectrum, color=color, alpha=0.6, lw=1.2,
                            label=f'_{class_name}')[0]  # 下划线防止重复

            # 记录图例
            if class_name not in legend_handles:
                legend_handles[class_name] = line
            neighbor_count += 1

    # 绘制中心像素（红色突出显示）
    center_class = data_gt[x, y]
    center_color = class_colors[center_class] if center_class != 0 else 'red'
    center_label = class_names[flag][center_class] if center_class != 0 else "Unknown"
    plt.plot(hypercube[x, y, :], color='red', lw=2, linestyle='--',
             label=f'Center: {center_label}')

    # 添加图例和标题
    handles = list(legend_handles.values()) + [plt.Line2D([0], [0], color='red', ls='--', lw=2)]
    labels = list(legend_handles.keys()) + [f'Center: {center_label}']

    # plt.title(f"Pixel ({x}, {y}) Spectrum - Center Class: {center_label}\n"
    #           f"Total Neighbors: {neighbor_count}", fontsize=14)
    plt.xlabel("Band Index", fontsize=12)
    plt.ylabel("Reflectance/Intensity", fontsize=12)
    plt.grid(alpha=0.3)

    # 分两列显示图例
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., ncol=2)

    # 保存图像
    filename = os.path.join(output_dir, f"{flag}_spectrum_{x}_{y}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"光谱图已保存至：{filename}")


if __name__ == "__main__":
    t = LoadingData()
    data, data_gt, h, w, c, class_nums = t.Loading('indian')

    for i in range(3, data.shape[0] - 2):
        for j in range(3, data.shape[1] - 2):
            s = set()
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    s.add(data_gt[x][y])

            if len(s) <= 3:
                continue

            plot_pixel_and_neighbors(data, data_gt, i, j, output_dir="neighbor_spectra", dpi=300)

    # plot_spectra_by_class(data, data_gt, flag='indian')
    # data_info(data_gt, data_gt, data_gt)
