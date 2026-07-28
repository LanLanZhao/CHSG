import os
import random
import time
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from numpy import unique
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from Config import ExperimentParams
from DataLoad import HyperData, applyPCA
from DrawHyper import get_map, get_class_detail
from HyperTools import LoadingData, ImageCut, split_data, train, test, \
    predict, split_data_fix, Timer, max_min_norm, mean_var_norm
from Model.CHSG import build_model, CAPE
from Utils.evaluation import HSIEvaluation
from Utils.scheduler import load_scheduler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

def LoadHyperspectral(args, random_state=42):
    """ 加载数据, 只要以 小patch 为单位的都可以这样加载 """
    print('\n... ... 正在加载高光谱的数据ing ... ...')
    Ld = LoadingData(args.data_root)
    data, labels, num_class, h, w, band = Ld.Loading(args.dataset)

    if args.norm == 'max_min_norm':
        data = max_min_norm(data)
    elif args.norm == 'mean_var_norm':
        data = mean_var_norm(data)

    map_labels = labels.copy()
    print(unique(map_labels))
    # 返回的是数据, 标签, 种类数, 图的长, 宽, 压维后的通道数
    # print(f'您正要训练的高光谱数据集为: {args.dataset}, 类别有 {num_class} 种 \n')

    if args.components != 0:
        print('\n... ... PCA(降维度) 转变 ... ...')
        data = applyPCA(data, numComponents=args.components)
        print('降维度后的高光谱形状为: ', data.shape)

    # print('高光谱图片处理后的维度形状为: ', data.shape, '\n', '正确的标签形状为: ', labels.shape)

    # print(f'\n以每一个像素点为中心形成宽高为 patch = {args.patch_size} 的立方块, 也称 pixels 块')

    # 为训练和测试创建数据集（只包含有标签的数据）
    pixels_train, labels_train = ImageCut(data, labels, window_size=args.patch_size, remove_zero_labels=True)

    # 为预测创建数据集（只包含有标签的数据，背景保持为黑色）
    pixels_all, labels_all = ImageCut(data, labels, window_size=args.patch_size, remove_zero_labels=True)

    # print('所有立方块的 X 的形状为: ', pixels.shape, '\n', '立方体的标签 y 为: ', labels.shape)

    # print('\n... ... 正在分割训练和测试数据 ... ...')

    if args.split_type == 'fixed':
        x_train, x_test, y_train, y_test = split_data_fix(pixels_train, labels_train, args.train_num, random_state=random_state)
    else:
        x_train, x_test, y_train, y_test = split_data(pixels_train, labels_train, args.train_ratio, random_state=random_state)

    print('X_train shape:', x_train.shape, '\n', 'X_test shape:', x_test.shape)

    # print(
    #     '\n为了适应 keras 结构，数据要做 transpose, 改变后的形状为(B * Depth * c * h * w), 为什么加个Depth ? 因为三维卷积要深度维')
    # x_train = rearrange(x_train, 'b h w c -> b 1 c h w')
    # x_test = rearrange(x_test, 'b h w c -> b 1 c h w')

    x_train = rearrange(x_train, 'b h w c -> b c h w')
    x_test = rearrange(x_test, 'b h w c -> b c h w')
    pixels_all = rearrange(pixels_all, 'b h w c -> b c h w')

    print('after transpose: x_train shape: ', x_train.shape)
    print('after transpose: x_test  shape: ', x_test.shape, '\n')

    # 封装好三个dataset
    train_dataset = HyperData(x_train, y_train, enhance=Args.data_aug)
    test_dataset = HyperData(x_test, y_test)
    all_dataset = HyperData(pixels_all, labels_all)

    TrainLoader = DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True, num_workers=Args.num_workers
                             , generator=torch.Generator().manual_seed(random_state))
    TestLoader = DataLoader(test_dataset, batch_size=Args.batch_size, shuffle=False, num_workers=Args.num_workers
                            , generator=torch.Generator().manual_seed(random_state))
    AllLoader = DataLoader(all_dataset, batch_size=Args.batch_size, shuffle=False, num_workers=Args.num_workers
                           , generator=torch.Generator().manual_seed(random_state))
    print('\n ... ... 数据预处理加载完毕  ... ...')

    # 训练迭代器, 测试迭代器, 全部的迭代器(画预测图用的), 以及数据集的真实标签(画True图用的), 降维度后的
    return TrainLoader, TestLoader, AllLoader, map_labels, band, num_class


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tsne_features(model, data_loader, device):
    """获取模型特征并进行t-SNE可视化"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            # 获取倒数第二层的特征
            if hasattr(model, 'get_features'):
                batch_features = model.get_features(batch_data)
            else:
                # 如果模型没有专门的特征提取方法,则使用最后一层的输入作为特征
                batch_features = model(batch_data)
                if isinstance(batch_features, tuple):
                    batch_features = batch_features[0]

            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    return features_2d, labels


# 添加可视化CAPE权重的函数
def visualize_cape_weights(model, save_path):
    """
    可视化CHSG模型中CAPE模块的权重热力图（分别保存RGB原图、patch图、热力图，无文字）

    Args:
        model: 训练好的CHSG模型
        save_path: 保存权重热力图的路径
    """
    print("\n[可视化] 开始可视化CAPE模块权重...")
    cape_vis_dir = os.path.join(save_path, 'cape_weights')
    os.makedirs(cape_vis_dir, exist_ok=True)

    # 生成单独的colorbar图像
    generate_standalone_colorbar(os.path.join(cape_vis_dir, 'colorbar.png'))

    cape_modules, cape_names = [], []

    def find_cape_modules(module, prefix=""):
        for name, submodule in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(submodule, CAPE):
                cape_modules.append(submodule)
                cape_names.append(full_name)
            else:
                find_cape_modules(submodule, full_name)

    find_cape_modules(model)

    if not cape_modules:
        print("[警告] 未检测到CAPE模块，检查模型结构")
        return

    print(f"[信息] 检测到 {len(cape_modules)} 个CAPE模块")

    first_stage_modules, first_stage_names = [], []
    for i, (mod, name) in enumerate(zip(cape_modules, cape_names)):
        if any(tag in name for tag in ["encoder.0", "block.0", "stage.0"]) or i == 0:
            first_stage_modules.append(mod)
            first_stage_names.append(name)

    if not first_stage_modules:
        first_stage_modules, first_stage_names = [cape_modules[0]], [cape_names[0]]

    print(f"[信息] 将可视化 {len(first_stage_modules)} 个第一阶段CAPE模块")

    try:
        Ld = LoadingData()
        data, labels, _, h, w, bands = Ld.Loading(Args.dataset)

        rgb_idx = [29, 19, 9]
        rgb_idx = [min(idx, bands - 1) for idx in rgb_idx]

        rgb_image = np.stack([data[:, :, i] for i in rgb_idx], axis=2)
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)

        margin = Args.patch_size // 2
        cx, cy = 100, 59  # 123, 20 .  100, 59.  44, 29.
        cx = max(margin, min(h - margin - 1, cx))
        cy = max(margin, min(w - margin - 1, cy))
        cls = labels[cx, cy]

        target_cls = 1
        cls_indices = np.where(labels == target_cls)
        if cls_indices[0].size > 0:
            idx = np.random.choice(cls_indices[0].shape[0])
            alt_x, alt_y = cls_indices[0][idx], cls_indices[1][idx]
            # 可启用：cx, cy = alt_x, alt_y

        print(f"[信息] 使用像素 ({cx}, {cy})，类别为 {cls}")

        xs, xe, ys, ye = cx - margin, cx + margin + 1, cy - margin, cy + margin + 1
        patch = data[xs:xe, ys:ye, :]

        if Args.norm == 'max_min_norm':
            patch = max_min_norm(patch)
        elif Args.norm == 'mean_var_norm':
            patch = mean_var_norm(patch)

        if Args.components:
            patch = data[xs:xe, ys:ye, :]

        patch = rearrange(patch, 'h w c -> 1 c h w')
        patch_tensor = torch.tensor(patch, dtype=torch.float32).to(device)

        original_patch = rgb_image[xs:xe, ys:ye]

        for i, (mod, name) in enumerate(zip(first_stage_modules, first_stage_names)):
            lambda1, lambda2 = mod.lamda1.item(), mod.lamda2.item()

            combined_output = None

            def hook_fn(module, input, output):
                nonlocal combined_output
                x = input[0]
                B, C, H, W = x.shape
                pe_out = module.pe(x)
                x_mid = x[:, :, H // 2: H // 2 + 1, W // 2: W // 2 + 1].repeat(1, 1, H, W)
                center = torch.sigmoid(-((x - x_mid).abs() ** 2)) * 2
                combined = module.lamda1 * pe_out + module.lamda2 * center
                combined_output = combined.detach().cpu()

            hook = mod.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(patch_tensor)
            hook.remove()

            if combined_output is not None:
                avg_combined = combined_output.mean(dim=1)[0]

                # 1. 保存 RGB 图（加红框）
                fig = plt.figure(figsize=(6, 6))
                plt.imshow(rgb_image)
                rect = plt.Rectangle((ys, xs), Args.patch_size, Args.patch_size,
                                     edgecolor='red', facecolor='none', linewidth=2)
                plt.gca().add_patch(rect)
                plt.axis('off')
                vis_path_rgb = os.path.join(cape_vis_dir, f'{name.replace(".", "_")}_rgb.png')
                plt.savefig(vis_path_rgb, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                # 2. 保存 Patch 图
                fig = plt.figure(figsize=(4, 4))
                plt.imshow(original_patch)
                plt.axis('off')
                vis_path_patch = os.path.join(cape_vis_dir, f'{name.replace(".", "_")}_patch.png')
                plt.savefig(vis_path_patch, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                # 3. 保存 热力图（添加 colorbar，固定颜色范围 0-1）
                # 3. 保存 热力图（自定义颜色条，固定 0-1）
                from matplotlib.colors import LinearSegmentedColormap

                fig, ax = plt.subplots(figsize=(6, 6))

                # 自定义颜色映射
                colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 0)]  # 白->黄->红->黑
                custom_cmap = LinearSegmentedColormap.from_list('custom_heatmap', colors, N=256)

                im = ax.imshow(avg_combined, cmap=custom_cmap, vmin=0.0, vmax=0.9)
                ax.axis('off')

                # 添加 colorbar
                # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # cbar.ax.tick_params(labelsize=8)

                vis_path_heatmap = os.path.join(cape_vis_dir, f'{name.replace(".", "_")}_heatmap.png')
                plt.savefig(vis_path_heatmap, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                print(f"[完成] 保存 {name} 图像到文件")
                colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 0)]
                custom_cmap = LinearSegmentedColormap.from_list('custom_heatmap', colors, N=256)

                # 2. 创建一个假数据
                gradient = np.linspace(0, 1, 256).reshape(-1, 1)

                # 3. 画出并只保存 colorbar
                fig, ax = plt.subplots(figsize=(1, 6))  # 宽度窄，高度高，像温度计

                # 画一个假的 imshow 用于生成 colorbar
                im = ax.imshow(gradient, aspect='auto', cmap=custom_cmap, origin='lower')

                # 去掉坐标轴
                ax.axis('off')

                # 添加 colorbar
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.5)
                cbar.ax.tick_params(labelsize=12)  # 可以改刻度字体大小

                # 保存
                plt.savefig("colorbar_only.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.close()

                print("已保存单独的温度条 colorbar！")

    except Exception as e:
        print(f"[错误] 可视化过程出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"[完成] 所有CAPE模块图像保存在: {cape_vis_dir}")


def generate_standalone_colorbar(save_path):
    """
    生成并保存单独的colorbar图像

    Args:
        save_path: 保存colorbar图像的路径
    """
    print("\n[可视化] 生成单独的colorbar...")

    # 创建自定义colormap - 颜色顺序需要反转以匹配水平条形图（左边0，右边1）
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.2, 0, 0), (0.4, 0, 0), (0.6, 0, 0), (0.8, 0, 0), (1, 0, 0),
              (1, 0.4, 0), (1, 0.6, 0), (1, 0.8, 0), (1, 1, 0.8), (1, 1, 1)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_heatmap', colors, N=256)

    # 创建一个水平渐变数据
    gradient = np.linspace(0, 1, 256).reshape(1, 256)

    # 创建图形 - 水平方向
    fig, ax = plt.subplots(figsize=(8, 1))

    # 隐藏主轴
    ax.set_visible(False)

    # 创建水平colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap),
                        ax=ax, orientation='horizontal',
                        fraction=1.0, pad=0.0)

    # 设置刻度标签
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    cbar.ax.tick_params(labelsize=12)

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.close()

    print(f"[完成] 单独的水平colorbar已保存到: {save_path}")


if __name__ == '__main__':
    Args = ExperimentParams()
    if type(Args.random_state) is int:
        seed_list = [Args.random_state]
    else:
        seed_list = Args.random_state
    experiment_num = Args.Experiment_num

    if len(seed_list) < experiment_num:
        extended_seed_list = []
        for i in range(experiment_num):
            extended_seed_list.append(seed_list[i % len(seed_list)])
        seed_list = extended_seed_list
    else:
        seed_list = seed_list[:experiment_num]

    StorageLocation = Args.output_dir

    if not os.path.isdir(StorageLocation):
        os.makedirs(StorageLocation)

    now_class = Args.dataset

    ok = False
    real_experiment_number = 1

    num_classes = len(get_class_detail(now_class))  # 建excel表格要用
    Experiment_result = np.zeros([num_classes + 6, Args.Experiment_num + 2])

    for count in range(1, Args.Experiment_num + 1):  # 实验的次数
        now_seed = seed_list[count - 1]
        seed_everything(now_seed)
        CountLocation = os.path.join(StorageLocation, Args.dataset, Args.model,
                                     "Experiment_" + f"{Args.dataset}_" + str(count))  # 存档当前的实验结果
        if not os.path.isdir(CountLocation):
            os.makedirs(CountLocation)
        train_loader, test_loader, all_loader, gt_labels, n_bands, categories = LoadHyperspectral(Args,
                                                                                                  random_state=now_seed)
        now_class = Args.dataset

        if Args.components != 0:
            n_bands = Args.components
        print(f'````````````````````第 {count} 次实验训练集测试数据已经预处理完成 并且加载完成 ```````````````````````')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = build_model(
            num_classes=num_classes,
            band=n_bands,
            patch_size=Args.patch_size,
        ).to(device)
        print("使用模型: CHSG")
        model.eval()

        # 获取一个批次的数据来获取正确的输入形状
        sample_batch = next(iter(train_loader))
        # 创建批次大小为2的输入样本
        dummy_input = sample_batch[0][0].to(device)

        # 计算和显示模型参数量
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"模型总参数量: {num_params:,}")
        print(f"可训练参数量: {num_trainable_params:,}")
        print(f"参数量(百万): {num_params / 1e6:.2f}M")

        dummy_input, _ = next(iter(train_loader))
        dummy_input = dummy_input[0:1].to(device)

        try:
            flops = FlopCountAnalysis(model, dummy_input)
            print(f"FLOPs: {flops.total() / 1e9:.3f} G")
        except Exception as e:
            print(f"[Warning] FLOPs calculation on {device} failed: {e}")
            print("[Warning] Retry FLOPs on CPU...")
            try:
                model = model.to("cpu")
                dummy_input_cpu = dummy_input.cpu()
                flops = FlopCountAnalysis(model, dummy_input_cpu)
                print(f"FLOPs (CPU fallback): {flops.total() / 1e9:.3f} G")
            except Exception as e_cpu:
                print(f"[Warning] FLOPs calculation skipped: {e_cpu}")
            finally:
                model = model.to(device)

        runtime1 = time.time()
        # ------------------------------------------------- 训练 -------------------------------------------
        print(f'\n............................正在进行训练第 {count} 次实验.........................')
        criterion = CrossEntropyLoss()

        # 初始化优化器
        optimizer, scheduler = load_scheduler(Args.model, model, Args)
        # 记录训练过程中的最优解
        best_loss = 9999999999

        Time = Timer()
        Time.start()
        train_acc_img, train_loss_img = [], []  # 记录所有loss 函数曲线和 精度 acc 的曲线

        for epoch in range(Args.epochs):
            epoch_start_time = time.time()  # 记录训练一个epoch的时间
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
            train_acc_img.append(train_acc)
            train_loss_img.append(train_loss)
            epoch_over_time = time.time()

            # save model
            print(f"Experiment [ {count} ]: [ Train | {epoch + 1:03d}/{Args.epochs:03d} ] loss = {train_loss:.5f},"
                  f" acc = {train_acc:.5f}", f"cost {epoch_over_time - epoch_start_time:.4f} s to train")
            if scheduler is not None:
                scheduler.step()

            # 保存当前最佳模型
            # if train_loss < best_loss:
            #     best_loss = train_loss
            #     torch.save(model.state_dict(), CountLocation + '/best_model.pth')

            # if epoch == 0:
            #     continue
            # if epoch % 40 == 0:
            #     test_loss, test_acc = test(test_loader, model, criterion, device)

        train_cost = Time.stop()

        # ````````````` Save output `loss, acc` as img ```````````
        plt.figure(1)
        plt.plot(np.array(train_loss_img), label='Training')
        plt.legend()
        plt.savefig(CountLocation + f'/Experiment_{count}_' + str(Args.dataset) + '_train_loss' + '.png')
        # plt.show()
        plt.figure(2)
        plt.plot(np.array(train_acc_img), label='Training')
        plt.legend()
        plt.savefig(CountLocation + f'/Experiment_{count}_' + str(Args.dataset) + '_train_acc' + '.png')
        # plt.show()

        print('\nTraining complete. Time cost about :', train_cost, ' 秒')
        print(f'\nCongratulations !!! 第 {count} 次实验，模型训练阶段完成！\n')
        #
        # ------------------------------------------------- 测试 -------------------------------------------------

        Time.start()

        test_loss, test_acc = test(test_loader, model, criterion, device)
        test_cost = Time.stop()

        print("\nTest_loss_avg: ", test_loss, "Test_acc_avg", test_acc, f"cost {test_cost:.4f} to test")
        print(f"\nCongratulations !!!  第 {count} 次实验，模型测试阶段完成！")

        # 保存模型统计信息
        # save_model_stats(model, device, dummy_input, Args, train_cost, test_cost)

        y_pred, y_true = predict(test_loader, model, device)  # 获取预测的数据 和 真实标签
        evalator = HSIEvaluation(Args.dataset)

        # print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (
        # epoch + 1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_true.shape)))

        classification, oa, confusion, each_acc, aa, kappa = evalator.eval(y_true, y_pred)

        # 超参全排列时使用
        print(f"\nOverall accuracy: {oa:.2f}")
        print(f"Average accuracy: {aa:.2f}")
        print(f"Kappa: {kappa:.2f}")

        # 各类别精度
        accuracies = [0] * categories
        for category in range(0, categories):
            # 获取当前类别的索引
            category_indices = (y_true == category)

            # 计算当前类别的精度：正确预测数量 / 该类别总数量
            correct_predictions = np.sum((y_pred[category_indices] == y_true[category_indices]))
            total_predictions = np.sum(category_indices)
            accuracies[category] = correct_predictions / total_predictions * 100

        runtime2 = time.time()
        all_cost = runtime2 - runtime1
        print(f'\n 第 {count} 次实验，模型评估阶段完成！\n')

        # ---------------------------------------------- Output Excel---------------------------------------
        Experiment_result[0, count - 1] = oa  # OA
        Experiment_result[1, count - 1] = aa  # AA
        Experiment_result[2, count - 1] = kappa  # Kappa
        Experiment_result[3, count - 1] = train_cost  # 训练时间
        Experiment_result[4, count - 1] = test_cost  # 测试时间
        Experiment_result[5, count - 1] = all_cost  # 整个训练时间

        Experiment_result[6: (6 + categories), count - 1] = accuracies  # 各小类的精度

        # ---------------------------------------------- Output txt---------------------------------------

        file_name = CountLocation + f"/Experiment_{count}_{Args.dataset} _classification_report.txt"
        with open(file_name, 'w') as x_file:
            x_file.write('{} one_xperiment_time (s)'.format(all_cost))
            x_file.write('\n')
            x_file.write('{} Training_Time (s)'.format(train_cost))
            x_file.write('\n')
            x_file.write('{} Test_time (s)'.format(test_cost))
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(kappa))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(oa))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(aa))
            x_file.write('\n')
            x_file.write('{} Each accuracy (%)'.format(each_acc))
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))

        print(f'第{count}次实验，模型评估阶段完成，结果已保存\n')

        get_map(model, device, all_loader, gt_labels, Args.dataset, CountLocation)

        # 添加t-SNE可视化
        # print("\n正在生成t-SNE可视化...")
        # features_2d, feature_labels = get_tsne_features(model, test_loader, device)
        #
        # # 创建tsne保存目录
        # tsne_dir = os.path.join(StorageLocation, 'tsne')
        # if not os.path.exists(tsne_dir):
        #     os.makedirs(tsne_dir)

        # custom_colors = [
        #     "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4",
        #     "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff",
        #     "#aa6e28", "#fffac8", "#800000", "#aaffc3"
        # ]
        # custom_colors = [
        #     "#DB5680", "#DB56B2", "#D256DB", "#A056DB", "#6F56DB", "#566FDB",
        #     "#56A1DB", "#56D3DB", "#56DBB1", "#56DB7F", "#5FDB56", "#91DB56",
        #     "#C2DB56", "#DBC256", "#DB9056", "#DB5E56"
        # ]
        # custom_colors = [
        #     "#FF0000", "#FF4500", "#FF8C00", "#FFD700",  # 红→橙→金（暖极）
        #     "#00FF00", "#32CD32", "#7CFC00", "#ADFF2F",  # 绿→黄绿（过渡）
        #     "#0000FF", "#1E90FF", "#4169E1", "#8A2BE2",  # 蓝→紫（冷极）
        #     "#FF00FF", "#EE82EE", "#DA70D6", "#BA55D3"  # 品红→紫（增强对比）
        # ]
        # from matplotlib.colors import ListedColormap
        # cmap = ListedColormap(custom_colors)
        # # 绘制t-SNE图
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=feature_labels, cmap=cmap, alpha=0.6)
        #
        # # # 配置颜色条，使其离散化并使用1开始的数字标签
        # unique_labels_in_plot = np.sort(np.unique(feature_labels))
        # if len(unique_labels_in_plot) > 0: # 仅当存在标签时才创建颜色条
        #     num_unique_classes = len(unique_labels_in_plot)
        #     plot_boundaries = np.arange(num_unique_classes + 1) - 0.5
        #     plot_ticks = unique_labels_in_plot
        #     cbar = plt.colorbar(scatter, ticks=plot_ticks, boundaries=plot_boundaries)
        #     # 假设 feature_labels 是0索引的，我们将其显示为1索引
        #     cbar.ax.set_yticklabels([str(int(label) + 1) for label in unique_labels_in_plot])
        #
        # # plt.title(f't-SNE Visualization of {Args.model} Features on {Args.dataset} (OA={oa:.2f}%)')
        #
        # # 保存图像
        # plt.tight_layout()
        # save_path = os.path.join(tsne_dir, f'{Args.dataset}_{Args.model}_OA_{oa:.2f}.png')
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"t-SNE可视化已保存到: {save_path}")

        # 将当前实验结果添加到answer.xlsx文件中
        import os.path

        all_result = os.path.join(StorageLocation, 'all_model_result/')  # 存放训练的报告

        if not os.path.isdir(all_result):
            os.makedirs(all_result)
        # 修改answer文件名，加上模型名称
        answer_file = all_result + f'{Args.model}_answer.xlsx'

        # 获取当前实验的真实序号
        if os.path.exists(answer_file):
            if ok:
                continue
            try:
                existing_df = pd.read_excel(answer_file)
                if not existing_df.empty and '总实验序号' in existing_df.columns:
                    if not ok:
                        ok = True
                        real_experiment_number = int(existing_df['总实验序号'].max()) + 1

            except Exception as e:
                print(f"读取现有文件获取序号时出错: {e}")
                real_experiment_number = 1
                ok = True

        # 计算总训练样本数量
        total_train_samples = 0
        if Args.split_type == 'fixed':
            # 固定数量模式下，总训练样本数 = 每类样本数 × 类别数
            # 注意：某些类别可能样本数不足train_num，所以这是一个上限估计
            total_train_samples = Args.train_num * categories
        else:
            # 比例模式下，尝试多种方法获取训练样本数
            try:
                # 方法1：直接从数据加载器获取
                total_train_samples = len(train_loader.dataset)
            except Exception as e1:
                print(f"无法从数据加载器获取样本数: {e1}")
                try:
                    # 方法2：从数据集的标签获取
                    if hasattr(train_loader.dataset, 'labels'):
                        total_train_samples = len(train_loader.dataset.labels)
                    elif hasattr(train_loader.dataset, 'y_train'):
                        total_train_samples = len(train_loader.dataset.y_train)
                    else:
                        raise AttributeError("数据集没有labels或y_train属性")
                except Exception as e2:
                    print(f"无法从数据集属性获取样本数: {e2}")
                    try:
                        if 'y_true' in locals() and y_true is not None:
                            test_samples = len(y_true)
                            # 估算总样本数 = 测试样本数 / (1 - 训练比例)
                            if Args.train_ratio < 1:  # 防止除以0
                                total_samples_estimate = int(test_samples / (1 - Args.train_ratio))
                                total_train_samples = total_samples_estimate - test_samples
                            else:
                                total_train_samples = -1  # 无法估算
                        else:
                            raise ValueError("y_true不可用")
                    except Exception as e3:
                        print(f"无法估算样本数: {e3}")
                        # 方法4：使用一个批次的大小和批次数量来估算
                        try:
                            batch_size = Args.batch_size
                            num_batches = len(train_loader)
                            total_train_samples = batch_size * num_batches
                        except Exception as e4:
                            print(f"无法使用批次估算样本数: {e4}")
                            total_train_samples = -1

        # 记录一下实际使用的训练样本数
        print(f"总训练样本数: {total_train_samples}")

        # 获取北京时间
        from datetime import datetime, timedelta

        # 尝试使用pytz库获取北京时间
        try:
            import pytz

            # 获取当前UTC时间
            utc_now = datetime.utcnow()
            # 转换为北京时间（UTC+8）
            beijing_timezone = pytz.timezone('Asia/Shanghai')
            beijing_time = utc_now.replace(tzinfo=pytz.utc).astimezone(beijing_timezone)
            # 格式化为易读的字符串
            beijing_time_str = beijing_time.strftime('%Y年%m月%d日%H时%M分')
        except ImportError:
            # 如果pytz库不可用，尝试安装
            try:
                import subprocess

                print("pytz库不可用，正在尝试安装...")
                subprocess.check_call(['pip', 'install', 'pytz'])

                # 安装成功后重新导入
                import pytz

                utc_now = datetime.utcnow()
                beijing_timezone = pytz.timezone('Asia/Shanghai')
                beijing_time = utc_now.replace(tzinfo=pytz.utc).astimezone(beijing_timezone)
                beijing_time_str = beijing_time.strftime('%Y年%m月%d日%H时%M分')
                print("pytz库安装成功，已正确获取北京时间")
            except Exception as e:
                # 如果安装失败，使用简单的时区偏移
                print(f"安装pytz库失败: {e}")
                print("使用简单的时区偏移计算北京时间")
                beijing_time = datetime.utcnow() + timedelta(hours=8)
                beijing_time_str = beijing_time.strftime('%Y年%m月%d日%H时%M分')

        # 准备要保存的数据
        experiment_data = {
            '总实验序号': real_experiment_number,
            '单次的实验序号': count,
            '数据集': Args.dataset,
            '模型': Args.model,
            'Epoch次数': Args.epochs,
            'BatchSize': Args.batch_size,
            '学习率': Args.lr,
            '训练方式': Args.split_type,
            '训练数量/比例': Args.train_num if Args.split_type == 'fixed' else Args.train_ratio,
            '总训练样本数': total_train_samples,
            'PatchSize': Args.patch_size,
            '随机种子': now_seed,  # 当前实验的种子
            '数据增强': '是' if Args.data_aug else '否',  # 添加数据增强字段
            'OA': float(oa),
            'AA': float(aa),
            'Kappa': float(kappa),
            '完成时间': beijing_time_str,
            '训练时间(秒)': float(train_cost),
            '测试时间(秒)': float(test_cost),
            '总时间(秒)': float(all_cost),
        }

        # 添加每个类别的精度
        class_details = get_class_detail(Args.dataset)
        for i, class_name in enumerate(class_details):
            if i < len(accuracies):
                experiment_data[f'类别{i + 1}_{class_name}'] = float(accuracies[i])

        # 将数据转换为DataFrame
        new_row_df = pd.DataFrame([experiment_data])

        # 检查文件是否存在，如果存在则追加，否则创建新文件
        if os.path.exists(answer_file):
            try:
                existing_df = pd.read_excel(answer_file)
                # 合并现有数据和新数据
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            except Exception as e:
                print(f"读取现有文件时出错: {e}")
                # 如果文件存在但无法读取，则创建新文件
                updated_df = new_row_df
        else:
            updated_df = new_row_df

        # 保存更新后的数据
        try:
            updated_df.to_excel(answer_file, index=False)
            print(f"\n第 {count} 次实验结果已添加到 {answer_file}")
        except Exception as e:
            print(f"保存结果到Excel文件时出错: {e}")
            # 尝试保存为CSV作为备份
            backup_file = os.path.join(StorageLocation, f'{Args.model}_answer_backup.csv')
            updated_df.to_csv(backup_file, index=False)
            print(f"已将结果保存到备份文件: {backup_file}")

        # 可选：可视化CHSG中的CAPE权重
        # if Args.model == 'CHSG':
        #     visualize_cape_weights(model, CountLocation)

    # 对所有训练的总结 并保存到excel
    if Args.Experiment_num > 1:
        # 创建一个新的结果数组，使用object类型来存储混合类型的数据
        New_Experiment_result = np.empty([num_classes + 17, Args.Experiment_num + 2], dtype=object)  # 增加行数以适应数据增强
        New_Experiment_result.fill(np.nan)  # 初始化为NaN

        # 复制原有数据到新数组中（向下移动一行）
        New_Experiment_result[1:num_classes + 7, :] = Experiment_result

        # 添加种子信息到第一行
        New_Experiment_result[0, 0:Args.Experiment_num] = seed_list[0:Args.Experiment_num]

        # 计算平均值和标准差（仅对精度相关的行）
        # 对于需要计算的行（OA, AA, Kappa和各类别精度）
        accuracy_rows = [1, 2, 3] + list(range(7, num_classes + 7))  # OA, AA, Kappa和各类别的行索引
        for row in range(New_Experiment_result.shape[0]):
            if row in accuracy_rows:
                # 确保进行计算的数据是数值类型
                row_data = New_Experiment_result[row, 0:-2].astype(float)
                New_Experiment_result[row, -2] = np.mean(row_data)
                New_Experiment_result[row, -1] = np.std(row_data)
            else:
                # 对于种子、训练时间、测试时间、实验时间和超参数，设置为None
                New_Experiment_result[row, -2] = None
                New_Experiment_result[row, -1] = None

        # 在最后几行添加超参数信息
        param_start_idx = num_classes + 8  # 增加1，为空行留出位置
        # 在最后一列（每个实验）都填入相同的超参数值
        for col in range(Args.Experiment_num):
            New_Experiment_result[param_start_idx, col] = str(Args.model)
            New_Experiment_result[param_start_idx + 1, col] = str(Args.dataset)
            New_Experiment_result[param_start_idx + 2, col] = str(Args.epochs)
            New_Experiment_result[param_start_idx + 3, col] = str(Args.lr)
            New_Experiment_result[param_start_idx + 4, col] = str(Args.batch_size)
            New_Experiment_result[param_start_idx + 5, col] = str(Args.patch_size)
            New_Experiment_result[param_start_idx + 6, col] = str(Args.components)
            New_Experiment_result[
                param_start_idx + 7, col] = f"{Args.split_type}({Args.train_num if Args.split_type == 'fixed' else Args.train_ratio})"
            New_Experiment_result[param_start_idx + 8, col] = '是' if Args.data_aug else '否'  # 添加数据增强信息

        data_df = pd.DataFrame(New_Experiment_result)

        # 更新索引名称，添加种子行和超参数行
        data_df.index = ['随机种子', 'OA', 'AA', 'Kappa', '训练时间', '测试时间', '实验时间'] + \
                        get_class_detail(now_class) + \
                        [''] + \
                        ['模型', '数据集', 'Epoch数', '学习率', 'Batch Size', 'Patch Size', 'PCA Components',
                         '训练方式', '数据增强']

        data_df.columns = ['第' + str(i) + '次实验' for i in range(1, Args.Experiment_num + 1)] + ['平均精度/耗时',
                                                                                                   '精度标准差']
        loca = time.strftime('%Y-%m-%d')
        filename = (os.path.join(StorageLocation, Args.dataset, Args.model) + f'/{str(Args.dataset)}_' + str(Args.model)
                    + f'_Run({Args.Experiment_num})' +
                    f'_{repr(round(float(New_Experiment_result[1, -2]), 2))}_') + str(loca) + '.xlsx'

        data_df.to_excel(excel_writer=filename, sheet_name="All&Mean&Std")
