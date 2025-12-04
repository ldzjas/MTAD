import sys

sys.path.append("../")
import logging
from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import seed_everything
from networks.usad import UsadModel
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
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
    parser.add_argument("--expid", type=str, default="usad_SWAT")
    parser.add_argument("--gpu", type=int, default=9)
    args = vars(parser.parse_args())

    config_dir = args["config"]
    experiment_id = args["expid"]

    params = load_config(config_dir, experiment_id)
    set_logger(params, args)
    logging.info(print_to_json(params))

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        valid_ratio=params["valid_ratio"],
        dim=params["dim"],
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
        test_windows_label = windows["test_label"]

        model = UsadModel(
            w_size=train_windows.shape[1] * train_windows.shape[2],
            z_size=train_windows.shape[1] * params["hidden_size"],
            device=params["device"],
        )
        tt = TimeTracker(nb_epoch=params["nb_epoch"])

        tt.train_start()
        model.fit(
            windows_train=train_windows,
            windows_val=None,
            epochs=params["nb_epoch"],
            batch_size=params["batch_size"],
        )
        tt.train_end()

        train_anomaly_score = model.predict_prob(
            windows_test=train_windows,
            batch_size=params["batch_size"],
        )
        tt.test_start()
        anomaly_score, anomaly_label = model.predict_prob(
            windows_test=test_windows,
            batch_size=params["batch_size"],
            windows_label=test_windows_label,
        )
        tt.test_end()

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

    #     store_entity(
    #         params,
    #         entity,
    #         train_anomaly_score,
    #         anomaly_score,
    #         anomaly_label,
    #         time_tracker=tt.get_data(),
    #     )
    # evaluator.eval_exp(
    #     exp_folder=params["model_root"],
    #     entities=params["entities"],
    #     merge_folder=params["benchmark_dir"],
    #     extra_params=params,
    # )
