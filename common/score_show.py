import matplotlib.pyplot as plt
import numpy as np

def plot_anomaly_scores(test_anomaly_score, test_start, test_end, valid_anomaly_max, alpha, anomaly_range, gap_time,
                        model_name):
    """
    绘制异常得分曲线，并标记异常区间和阈值。

    Args:
        test_anomaly_score (np.ndarray): 测试数据的异常得分。
        test_start (int): 测试数据开始索引。
        test_end (int): 测试数据结束索引。
        valid_anomaly_max (float): 有效异常得分的最大值。
        alpha (float): 阈值乘数。
        anomaly_range (list): 异常区间列表，每个元素为 [start, end]。
        gap_time (int): 时间间隔。
        model_name (str): 模型名称，用于保存图片。
    """
    fig, axes = plt.subplots(figsize=(12, 6))  # 调整图像大小以获得更好的显示效果

    test_num = test_end - test_start

    # 绘制异常得分曲线
    axes.plot(test_anomaly_score, color='black', linewidth=2, label='Anomaly Score')

    # # 绘制阈值线
    # threshold = np.full((test_num), valid_anomaly_max * alpha)
    # axes.plot(threshold, color='blue', linestyle='--', linewidth=2, label='Threshold')  # 将阈值线颜色改为蓝色
    #
    # # 绘制异常范围
    # for k in range(len(anomaly_range)):
    #     axes.axvspan(anomaly_range[k][0] - 10 / gap_time, anomaly_range[k][1] - 10 / gap_time, color='red', alpha=0.3,
    #                  label='Anomaly Region' if k == 0 else "")  # 增加透明度，并只添加一次标签

    # 设置图表样式
    axes.set_xlabel('Test Time', fontsize=18)  # 调整字体大小
    axes.set_ylabel('Anomaly Score', fontsize=18)  # 调整字体大小
    axes.set_title(f'{model_name} Anomaly Detection', fontsize=20)  # 调整字体大小，并包含模型名称

    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')

    # 调整刻度标签字体大小
    axes.tick_params(axis='x', labelsize=14)
    axes.tick_params(axis='y', labelsize=14)

    # 添加图例
    axes.legend(fontsize=14)

    fig.subplots_adjust(bottom=0.15, left=0.15)  # 调整边距

    # 保存图片到 outputs 文件夹，文件名包含模型名称
    plt.savefig(f'./outputs/{model_name}_anomaly_scores.jpg', bbox_inches='tight')
    plt.show()
