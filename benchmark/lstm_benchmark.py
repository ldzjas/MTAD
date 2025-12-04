import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

sys.path.append("../")
import logging
from common import data_preprocess
from common.dataloader import get_dataloaders, load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
from networks.lstm import LSTM
from common.score_show import plot_anomaly_scores  # 添加导入语句
import numpy as np

seed_everything()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./benchmark_config/",
        help="The config directory.",
    )
    parser.add_argument("--expid", type=str, default="lstm_SWAT")
    parser.add_argument("--gpu", type=int, default=5)
    args = vars(parser.parse_args())

    config_dir = args["config"]
    experiment_id = args["expid"]

    params = load_config(config_dir, experiment_id)
    set_logger(params, args)
    logging.info(print_to_json(params))

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix1"],
        nrows=params["nrows"],
    )

    data_dict2 = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix2"],
        nrows=params["nrows"],
    )

    # # preprocessing
    # pp = data_preprocess.preprocessor(model_root=params["model_root"])
    # data_dict = pp.normalize(data_dict, method=params["normalize"])

    # sliding windows
    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    window_dict2 = data_preprocess.generate_windows(
        data_dict2,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    # train/test on each entity put here
    evaluator = Evaluator(**params["eval"])
    for entity in params["entities"]:
        logging.info("Fitting dataset: {}".format(entity))
        windows = window_dict[entity]
        windows2 = window_dict2[entity]
        train_windows = np.concatenate((windows["train_windows"], windows2["train_windows"]), axis=0)
        test_windows = windows["test_windows"]

        train_loader, _, test_loader = get_dataloaders(
            train_windows,
            test_windows,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
        )

        model = LSTM(
            in_channels=params["dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            window_size=params["window_size"],
            prediction_length=params["prediction_length"],
            prediction_dims=params["prediction_dims"],
            patience=params["patience"],
            save_path=params["model_root"],
            nb_epoch=params["nb_epoch"],
            lr=params["lr"],
            device=params["device"],
        )

        tt = TimeTracker(nb_epoch=params["nb_epoch"])
        tt.train_start()
        model.fit(
            train_loader,
            test_loader=test_loader,
            test_label=windows["test_label"],
        )
        tt.train_end()

        model.load_encoder()
        train_anomaly_score = model.predict_prob(train_loader)

        tt.test_start()
        anomaly_score, anomaly_label = model.predict_prob(
            test_loader, windows["test_label"]
        )
        tt.test_end()

        # 调用可视化函数
        # 假设 anomaly_range 和 gap_time 需要根据实际情况获取或定义
        # 这里使用示例值，您可能需要根据您的数据和需求进行调整
        # anomaly_range = [[2142,2201],[3043,3102],[3613,3672],[4514,4573],[4795,4854]] # 示例异常区间
        # gap_time = 1 # 示例时间间隔
        # model_name = experiment_id # 使用实验ID作为模型名称

        # 为了演示，我将使用您提供的 anomaly_range 和一个默认的 gap_time
        # 您需要根据实际情况获取这些值
        anomaly_range_example = [[2142, 2201], [3043, 3102], [3613, 3672], [4514, 4573], [4795, 4854]]
        gap_time_example = 1  # 假设每个时间步长为1

        plot_anomaly_scores(
            anomaly_score,
            0,  # test_start 假设从0开始
            len(anomaly_score),  # test_end 假设为异常得分的长度
            np.max(anomaly_score),  # valid_anomaly_max 可以是异常得分的最大值
            0.5,  # alpha 阈值乘数，可以根据需要调整
            anomaly_range_example,
            gap_time_example,
            experiment_id.split('_')[0]  # 使用实验ID中第一个下划线之前的部分作为模型名称
        )

        # store_entity(
        #     params,
        #     entity,
        #     train_anomaly_score,
        #     anomaly_score,
        #     anomaly_label,
        #     time_tracker=tt.get_data(),
        # )
    # evaluator.eval_exp(
    #     exp_folder=params["model_root"],
    #     entities=params["entities"],
    #     merge_folder=params["benchmark_dir"],
    #     extra_params=params,
    # )
