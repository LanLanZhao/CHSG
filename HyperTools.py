import os

import hdf5storage
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.io as sio
import time


class LoadingData:
    """
        This module is used to load the basic data of the data set
        __init__() : data dir root
        Loading() : Please choose what Hyperspectral data you want to load
    """

    def __init__(self, base_path='./dataset'):
        self.base_path = os.path.abspath(os.path.expanduser(base_path))
        self.data_info = {
            'indian': ('indian_pines_corrected.mat', 'indian_pines_gt.mat', 'indian_pines_corrected', 'indian_pines_gt')
            ,
            'paviau': ('PaviaU.mat', 'PaviaU_gt.mat', 'paviaU', 'paviaU_gt')
            ,
            'ksc': ('KSC_corrected.mat', 'KSC_gt.mat', 'KSC', 'KSC_gt')
            ,
            'sali': ('Salinas_corrected.mat', 'Salinas_gt.mat', 'salinas_corrected', 'salinas_gt')
            ,
            'botswana': ('Botswana.mat', 'Botswana_gt.mat', 'Botswana', 'Botswana_gt')
            ,
            'houston': ('Houston13.mat', 'Houston13_gt.mat', 'Houston', 'Houston_gt')
            ,
            'hanchuan': ('WHU_Hi_HanChuan.mat', 'WHU_Hi_HanChuan_gt.mat', 'WHU_Hi_HanChuan', 'WHU_Hi_HanChuan_gt')
            ,
            'honghu': ('WHU_Hi_HongHu.mat', 'WHU_Hi_HongHu_gt.mat', 'WHU_Hi_HongHu', 'WHU_Hi_HongHu_gt')
            ,
            'longkou': ('WHU_Hi_LongKou.mat', 'WHU_Hi_LongKou_gt.mat', 'WHU_Hi_LongKou', 'WHU_Hi_LongKou_gt')
            ,
            'houston2018': ('2018Houston.mat', '2018Houston_gt.mat', 'houstonU', 'houstonU_gt')
            ,
            'paviac': ('Pavia.mat', 'Pavia_gt.mat', 'pavia', 'pavia_gt')
            ,
            'SZUR1': ('SZUTreeHSI_R1.mat', 'SZUTreeHSI_R1_gt.mat', 'hyperspectral_data_98bands', 'ndvi_label')
            ,
            'SZUR2': ('SZUTreeHSI_R2.mat', 'SZUTreeHSI_R2_gt.mat', 'hyperspectral_data_98bands', 'ndvi_label')
            ,
            'UP': ('Utopia.mat', 'Utopia_gt.mat', 'Utopia', 'Utopia_gt')
            ,
            'HC': ('holden.mat', 'holden_gt.mat', 'holden', 'holden_gt')
            ,
            'NF': ('NiliFossae.mat', 'NiliFossae_gt.mat', 'NiliFossae', 'NiliFossae_gt')
            ,
            'loukia': ('Loukia.mat', 'Loukia_gt.mat', 'ori_data', 'map'),

            'dioni': ('Dioni.mat', 'Dioni_gt_out68.mat', 'ori_data', 'map'),

            'tea': ('tea.mat', 'tea_gt.mat', 'tea', 'tea_gt'),

            'xuzhou': ('xuzhou.mat', 'xuzhou_gt.mat', 'xuzhou', 'xuzhou_gt'),

            'chi': ('Chikusei.mat', 'Chikusei_gt.mat', 'chikusei', 'GT')
        }

    def Loading(self, name='indian'):
        if name not in self.data_info:
            raise ValueError("Invalid dataset flag provided.")

        data_path = os.path.join(self.base_path, self.data_info[name][0])
        label_path = os.path.join(self.base_path, self.data_info[name][1])
        missing = [path for path in (data_path, label_path) if not os.path.isfile(path)]
        if missing:
            missing_list = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                f"Dataset '{name}' is incomplete. Missing files:\n{missing_list}\n"
                "Set the dataset directory with --data-root."
            )

        dt1 = hdf5storage.loadmat(data_path)
        dt2 = hdf5storage.loadmat(label_path)

        print(dt1.keys())
        print(dt2.keys())

        data = dt1[self.data_info[name][2]]
        labels = dt2[self.data_info[name][3]]

        print(set(np.unique(labels)))
        print(data.shape)
        h, w, c = data.shape
        if name == 'tea' or name == 'xuzhou':
            labels = labels.reshape(h, w)

        # if name == 'hanchuan' or name == 'honghu' or name == 'longkou':
            # data = data.astype(np.int64)

        num_class = len(np.unique(labels)) - 1
        return data, labels, num_class, h, w, c


def mean_var_norm(data):
    """均值方差归一化"""
    mean = np.mean(data, axis=(0, 1), keepdims=True)  # 计算均值
    std = np.std(data, axis=(0, 1), keepdims=True)  # 计算标准差
    return (data - mean) / (std + 1e-8)  # 防止除零


def max_min_norm(data):
    """ 最大最小归一化 """
    norm_data = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:, :, i])
        input_min = np.min(data[:, :, i])
        norm_data[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min)
    return norm_data


def ImageCut(X, y, window_size=11, remove_zero_labels=True):
    """
    从输入图像 X 中提取每个像素周围的 patch，并与对应的标签 y 结合形成符合 Keras 处理的数据格式。

    参数:
    X: 输入图像，形状为 (height, width, channels)
    y: 标签矩阵，形状为 (height, width)
    window_size: 提取的 patch 大小，必须为奇数 (默认为 11)
    remove_zero_labels: 是否移除标签为 0 的 patch (默认为 True)

    返回:
    patches_data: 提取的 patch 数据，形状为 (num_patches, window_size, window_size, channels)
    patches_labels: 对应的标签，形状为 (num_patches,)
    """
    # 计算填充的边界大小
    margin = window_size // 2
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

    if remove_zero_labels:
        # 获取非零标签的位置
        valid_positions = np.where(y > 0)
    else:
        # 获取所有位置（包括标签为0的点）
        valid_positions = np.where(y >= 0)

    num_patches = len(valid_positions[0])

    # 初始化 patch 数据和标签
    patches_data = np.zeros((num_patches, window_size, window_size, X.shape[2]))
    patches_labels = np.zeros(num_patches)

    # 遍历所有有效位置
    for idx, (i, j) in enumerate(zip(valid_positions[0], valid_positions[1])):
        # 注意：i和j已经是原始图像中的位置，需要加上margin来获取padded图像中的位置
        patch = padded_X[i:i + window_size, j:j + window_size]
        patches_data[idx] = patch
        patches_labels[idx] = y[i, j]

    # 处理标签：如果remove_zero_labels为True，将非零标签减1使其从0开始
    # 如果remove_zero_labels为False，保持原样，但确保0标签点在预测时被正确处理
    if remove_zero_labels:
        # 将非零标签减1，使其从0开始
        non_zero_mask = patches_labels > 0
        patches_labels[non_zero_mask] = patches_labels[non_zero_mask] - 1

    return patches_data, patches_labels


def random_unison(a, b, random_state=None):
    """
    Shuffles two arrays in unison, maintaining the alignment of elements between them.

    Parameters:
        a (np.array): First array to shuffle.
        b (np.array): Second array to shuffle, aligned with 'a'.
        random_state (int, optional): Seed for reproducibility. Default is None.

    Returns:
        tuple: Shuffled versions of input arrays a and b.
    """
    assert len(a) == len(b), "Input arrays must have the same length."
    rng = np.random.default_rng(seed=random_state)
    perm = rng.permutation(len(a))
    return a[perm], b[perm]


def split_data_fix(pixels, labels, n_samples, random_state=None):
    """
    Splits the dataset into training and testing sets with a fixed number of samples per class.

    Parameters:
    - pixels (ndarray): Pixel data of shape (N, H, W, C).
    - labels (ndarray): Labels for each sample, shape (N,).
    - n_samples (int): Number of samples to use for training per class.
    - random_state (int or None): Random seed for reproducibility.

    Returns:
    - train_x (ndarray): Training pixel data, shape (total_train_size, H, W, C).
    - test_x (ndarray): Testing pixel data, shape (total_test_size, H, W, C).
    - train_y (ndarray): Training labels, shape (total_train_size,).
    - test_y (ndarray): Testing labels, shape (total_test_size,).
    """
    # Get unique classes and their pixel counts
    unique_classes, pixel_counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_classes)

    # 计算每个类应该取的实际样本数
    actual_samples = []
    for count in pixel_counts:
        if count > n_samples:  # 类别样本数量足够
            actual_samples.append(n_samples)
        elif count > 50:  # 样本数不够n_samples，但数量较多
            actual_samples.append(50)
        else:  # 样本数量较少
            actual_samples.append(count)

    total_train_size = sum(actual_samples)
    total_test_size = len(labels) - total_train_size

    # Preallocate memory for training and testing sets
    train_x = np.empty((total_train_size, *pixels.shape[1:]), dtype=pixels.dtype)
    train_y = np.empty(total_train_size, dtype=labels.dtype)
    test_x = np.empty((total_test_size, *pixels.shape[1:]), dtype=pixels.dtype)
    test_y = np.empty(total_test_size, dtype=labels.dtype)

    train_index = 0
    test_index = 0

    rng = np.random.default_rng(random_state)

    # 对每个种类抽取训练的数量
    for idx, cls in enumerate(unique_classes):
        class_indices = np.where(labels == cls)[0]
        rng.shuffle(class_indices)  # Shuffle indices

        # 使用当前类别的实际样本数量
        samples_for_this_class = actual_samples[idx]
        train_indices = class_indices[:samples_for_this_class]
        test_indices = class_indices[samples_for_this_class:]

        print("Class ", cls, " all_counts:", len(class_indices), f"  for trains_counts: {len(train_indices)}   ",
              f"for test_counts: {len(test_indices)}")
        # Add to training set
        train_end = train_index + len(train_indices)
        train_x[train_index: train_end] = pixels[train_indices]
        train_y[train_index: train_end] = labels[train_indices]
        train_index = train_end

        # Add to testing set
        test_end = test_index + len(test_indices)
        test_x[test_index: test_end] = pixels[test_indices]
        test_y[test_index: test_end] = labels[test_indices]
        test_index = test_end

    # Shuffle the training set for randomness
    train_x, train_y = random_unison(train_x, train_y, rng)
    return train_x, test_x, train_y, test_y


def split_data(pixels, labels, train_percent, method="custom", random_state=None):
    """
    Splits data into training and testing sets using either sklearn's split or a custom implementation.
    custom : Custom according to the training scale, here is custom is to train each small class to scale,
    rather than all the classes mixed together to scale

    Parameters:
        pixels (np.array): Array of pixel data.
        labels (np.array): Array of corresponding labels.
        train_percent (float): Proportion of data to include in the training set.
        method (str): "sklearn" to use train_test_split, or "custom" for a manual split.
        random_state (int, optional): Seed for reproducibility. Default is 69.

    Returns:
        tuple: Training and testing sets for both pixels and labels.
    """
    if method == "sklearn":
        return train_test_split(
            pixels, labels, test_size=(1 - train_percent), stratify=labels, random_state=random_state
        )
    elif method == "custom":
        unique_labels, counts = np.unique(labels, return_counts=True)
        train_counts = np.ceil(counts * train_percent).astype(int)
        train_pixels, train_labels, test_pixels, test_labels = [], [], [], []

        for idx, (label, train_count) in enumerate(zip(unique_labels, train_counts)):
            label_pixels = pixels[labels == label]
            label_labels = labels[labels == label]

            print("Class ", int(idx), " all_nums:", len(label_labels), f" for train_counts: {train_count}   ",
                  f"for test_counts:   {len(label_labels) - train_count}")

            shuffled_pixels, shuffled_labels = random_unison(label_pixels, label_labels, random_state=random_state)
            train_pixels.append(shuffled_pixels[:train_count])
            train_labels.append(shuffled_labels[:train_count])
            test_pixels.append(shuffled_pixels[train_count:])
            test_labels.append(shuffled_labels[train_count:])

        # Concatenate lists into arrays
        train_pixels = np.concatenate(train_pixels)
        train_labels = np.concatenate(train_labels)
        test_pixels = np.concatenate(test_pixels)
        test_labels = np.concatenate(test_labels)

        # Final shuffle of training data
        train_pixels, train_labels = random_unison(train_pixels, train_labels, random_state=random_state)

        return train_pixels, test_pixels, train_labels, test_labels
    else:
        raise ValueError("Method must be either 'sklearn' or 'custom'.")


def SplitData(pixels, labels, class_num, train_ratio=0.1, val_ratio=0.1, train_num=10, val_num=10,
              samples_type='ratio', random_state=None, global_shuffle=False, flag='indian'):
    """
    划分数据集为训练、验证、测试集，返回像素和标签，支持比例或固定数量模式。

    Parameters:
    -----------
    pixels : np.ndarray
        像素数据，形状为 (N, H, W, C)。
    labels : np.ndarray
        标签数据，形状为 (N,)。
    class_num : int
        总类别数（假设标签为 0-based）。
    train_ratio : float
        训练集比例（仅当 samples_type='ratio' 生效）。
    val_ratio : float
        验证集比例（仅当 samples_type='ratio' 生效）。
    train_num : int
        每类训练集固定数量（仅当 samples_type='fixed' 生效）。
    val_num : int
        每类验证集固定数量（仅当 samples_type='fixed' 生效）。
    samples_type : str
        划分模式：'ratio'（按比例）或 'fixed'（按固定数量）。
    random_state : int
        随机种子。
    global_shuffle : bool
        是否在合并后全局打乱数据。

    Returns:
    --------
    (train_pixels, val_pixels, test_pixels), (train_labels, val_labels, test_labels)
        训练、验证、测试集的像素和标签。
    """
    # 初始化存储容器
    train_pixels, val_pixels, test_pixels = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    # 按类别划分
    for cls in range(class_num):
        # 获取当前类别的索引
        cls_mask = (labels == cls)
        cls_pixels = pixels[cls_mask]
        cls_labels = labels[cls_mask]
        num_samples = len(cls_labels)

        if num_samples == 0:
            print(f"Class {cls}: 无样本，跳过。")
            continue

        # 打乱当前类别的数据
        np.random.seed(random_state)
        shuffle_idx = np.random.permutation(num_samples)
        cls_pixels = cls_pixels[shuffle_idx]
        cls_labels = cls_labels[shuffle_idx]

        # 计算划分数量
        if samples_type == 'ratio':
            train_size = round(num_samples * train_ratio)
            val_size = round(num_samples * val_ratio)

            # 确保 train + val <= 总样本数
            if train_size + val_size > num_samples:
                val_size = num_samples - train_size
        elif samples_type == 'fixed':
            if train_num is None or val_num is None:
                raise ValueError("固定数量模式需指定 train_num 和 val_num")
            train_size = min(train_num, num_samples)
            val_size = min(val_num, num_samples - train_size)
            if int(cls) == 8 and flag == 'indian':
                train_size = 10
                if val_num != 0:
                    val_size = 5
            if int(cls) == 6 and flag == 'indian':
                train_size = 10
                if train_num != 0:
                    val_size = 5
        else:
            raise ValueError("samples_type 必须为 'ratio' 或 'fixed'")

        # 划分数据
        train_cls_pixels = cls_pixels[:train_size]
        val_cls_pixels = cls_pixels[train_size:train_size + val_size]
        test_cls_pixels = cls_pixels[train_size + val_size:]

        train_cls_labels = cls_labels[:train_size]
        val_cls_labels = cls_labels[train_size:train_size + val_size]
        test_cls_labels = cls_labels[train_size + val_size:]

        # 记录划分结果
        train_pixels.append(train_cls_pixels)
        train_labels.append(train_cls_labels)
        val_pixels.append(val_cls_pixels)
        val_labels.append(val_cls_labels)
        test_pixels.append(test_cls_pixels)
        test_labels.append(test_cls_labels)

        # 打印信息
        print(
            f"Class {cls}: all_num = {num_samples} | train_num = {len(train_cls_labels)} |"
            f" valid_num = {len(val_cls_labels)} | test_num = {len(test_cls_labels)}")

    # 合并所有类别的数据
    def _concat(data_list):
        return np.concatenate(data_list, axis=0) if data_list else np.array([])

    train_pixels = _concat(train_pixels)
    train_labels = _concat(train_labels)
    val_pixels = _concat(val_pixels)
    val_labels = _concat(val_labels)
    test_pixels = _concat(test_pixels)
    test_labels = _concat(test_labels)

    # 全局打乱（可选）
    if global_shuffle:
        def _shuffle(data, Labels):
            if len(data) == 0:
                return data, Labels
            shuffled_idx = np.random.permutation(len(data))
            return data[shuffled_idx], Labels[shuffled_idx]

        np.random.seed(random_state)
        train_pixels, train_labels = _shuffle(train_pixels, train_labels)
        val_pixels, val_labels = _shuffle(val_pixels, val_labels)
        test_pixels, test_labels = _shuffle(test_pixels, test_labels)

    return (train_pixels, val_pixels, test_pixels), (train_labels, val_labels, test_labels)


def adjust_learning_rate(Optimizer, Epoch, args):
    """ Dynamically adjust the learning rate """
    lr = args.lr * (0.1 ** (Epoch // 150)) * (0.1 ** (Epoch // 225))
    for param_group in Optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          ):
    """训练函数改进版（保持简洁）

    主要优化点：
    1. 移除不必要的.cpu()操作，统一设备
    2. 使用累加统计代替列表存储，减少内存占用
    3. 添加类型提示提升可读性
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * images.size(0)
        total_correct += correct
        total_samples += images.size(0)

    # 计算平均指标
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def train_and_valid(TrainLoader, ValidLoader, Model, Criterion, Optimizer, Epoch, Device):
    Model.to(Device)

    timer = Timer()
    # ----------------------------------- Training --------------------------------------
    Model.train()
    metricT = Accumulator(3)  # 损失值, 正确的数量, 总数
    for i, (images, labels) in enumerate(TrainLoader):
        images, labels = images.to(Device), labels.to(Device)
        timer.start()
        Optimizer.zero_grad()
        logits = Model(images)
        # 我们不许要使用softmax因为求交叉熵的时候会自动done
        loss = Criterion(logits, labels)
        loss.backward()
        Optimizer.step()
        # 损失的总和， 正确的总数， 训练的数据总数
        acc = (logits.argmax(dim=-1) == labels).sum().item()
        metricT.add(loss.sum() * images.size(0), acc, labels.shape[0])
        timer.stop()

    # 训练集的平均损失和准确度的平均值
    train_loss = metricT[0] / metricT[2]
    train_acc = metricT[1] / metricT[2]

    # ----------------------------------- Validation ----------------------------------------
    Model.eval()
    metricV = Accumulator(3)

    for i, (images, labels) in enumerate(ValidLoader):
        images, labels = images.to(Device), labels.to(Device)
        with torch.no_grad():
            logits = Model(images)
            loss = Criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).sum().item()
            # 验证损失 正确数量 验证总数
            metricV.add(loss.sum() * images.size(0), acc, labels.shape[0])

    # 验证集集平均损失和准确度的平均值
    valid_loss = metricV[0] / metricV[2]
    valid_acc = metricV[1] / metricV[2]
    return train_loss, train_acc, valid_loss, valid_acc


def test(test_loader, model, criterion, device):
    """
    return ：
        test_avg_loss (float): 测试集平均损失
        test_accuracy (float): 测试集准确率
    """
    print('\n~~~~ Running Testing ~~~~~')
    model.eval()
    model.to(device)

    test_losses = []
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        # 使用tqdm显示进度条（可选）
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

            # 计算准确率
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    # 计算总体指标
    test_avg_loss = sum(test_losses) / len(test_losses)
    test_accuracy = total_correct / total_samples

    print(f'Test Loss: {test_avg_loss:.4f} | Test Acc: {test_accuracy * 100:.2f}%')
    return test_avg_loss, test_accuracy


def predict(TestLoader, Model, Device):
    """
    The function used to predict
    returns: the sequence of `predicted labels` and the sequence of `correct labels`
    """
    Model.eval()
    Ypred, Ytest = [], []
    for inputs, labels in TestLoader:
        inputs = inputs.to(Device)
        outputs = Model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        Ypred = np.concatenate((Ypred, outputs))
        Ytest = np.concatenate((Ytest, labels))
    return Ypred.astype('float32'), Ytest.astype('float32')


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
