import torch
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import numpy as np


class HyperData(Dataset):
    """
    自定义 Dataset 用于加载 patch 数据，并加入数据增强操作

    // ? 可以在这里对数据进行加强的预处理, 对小样本的增强很明显
    """

    def __init__(self, pixels, labels, enhance=False):
        """
        初始化数据集
        Args:
            pixels: 输入数据，形状为 (num_samples, channels, height, width)
            labels: 标签数据，形状为 (num_samples,)
            # flip_enhance: 是否启用随机翻转数据增强
            # radiation_enhance: 是否启用辐射噪声数据增强
            # mixture_enhance: 是否启用混合噪声数据增强
        """
        self.data = torch.FloatTensor(pixels)  # 转换为 PyTorch 张量
        self.labels = torch.LongTensor(labels)  # 转换为 PyTorch 张量

        self.flip_enhance = enhance
        self.rotation_enhance = enhance

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]  # 获取数据
        label = self.labels[idx]  # 获取标签

        # 将数据转换为 NumPy 数组以便进行数据增强
        data_np = data.numpy()

        # 数据增强操作
        if self.flip_enhance:
            data_np = self.flip(data_np)
        # if self.rotation_enhance:
        #     data_np = self.radiation_noise(data_np)
        # 确保数组的步幅是正的
        data_np = np.ascontiguousarray(data_np)

        # 将数据转换回 PyTorch 张量
        data = torch.FloatTensor(data_np)
        return data, label

    @staticmethod
    def flip(arrays):
        """
        随机翻转数据增强
        Args:
            arrays: 输入数据，形状为 (channels, height, width)
        Returns:
            翻转后的数据
        """
        horizontal = np.random.random() > 0.5  # 随机决定是否水平翻转
        vertical = np.random.random() > 0.5  # 随机决定是否垂直翻转
        if horizontal:
            arrays = np.flip(arrays, axis=-2)  # 水平翻转 (沿 width 轴)
        if vertical:
            arrays = np.flip(arrays, axis=-1)  # 垂直翻转 (沿 height 轴)
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        """
        辐射噪声数据增强
        Args:
            data: 输入数据，形状为 (channels, height, width)
            alpha_range: 随机缩放因子的范围
            beta: 噪声强度
        Returns:
            添加辐射噪声后的数据
        """
        alpha = np.random.uniform(*alpha_range)  # 随机缩放因子
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)  # 生成噪声
        return alpha * data + beta * noise

    def __labels__(self):
        """
        返回所有标签
        """
        return self.labels


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    newX = apply_savgol_to_hyperspectral(newX)
    return newX


def apply_savgol_to_hyperspectral(image, window_length=13, polyorder=2):
    """
    对高光谱图像的光谱维度应用 Savitzky-Golay 滤波。

    参数:
    - image: 输入的高光谱图像，形状为 (h, w, c)
    - window_length: 滤波窗口的长度，必须是奇数
    - polyorder: 多项式拟合的阶数

    返回:
    - filtered_image: 滤波后的高光谱图像，形状与输入图像相同
    """
    h, w, c = image.shape
    filtered_image = np.zeros_like(image)

    # 对每个像素的光谱曲线进行滤波
    for i in range(h):
        for j in range(w):
            filtered_image[i, j, :] = savgol_filter(image[i, j, :], window_length, polyorder)

    return filtered_image
