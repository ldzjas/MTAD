import sys

sys.path.append("../")
import logging
import argparse
from networks.anomaly_transformer.solver import AnomalyTransformer

from common import data_preprocess
from common.dataloader import get_dataloaders, load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
from common.score_show import plot_anomaly_scores  # 添加导入语句
import numpy as np

seed_everything()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./benchmark_config/",
        help="The config directory.",
    )
    parser.add_argument("--expid", type=str, default="anomaly_transformer_SWAT")
    parser.add_argument("--gpu", type=int, default=3)
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
        test_windows_label = windows["test_label"]

        train_loader, valid_loader, test_loader = get_dataloaders(
            train_windows,
            test_windows,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
        )

        model = AnomalyTransformer(
            lr=params["lr"],
            num_epochs=params["nb_epoch"],
            k=params["k"],
            win_size=params["window_size"],
            input_c=params["dim"],
            output_c=params["dim"],
            batch_size=params["batch_size"],
            model_save_path=params["model_root"],
            device=params["device"],
        )

        tt = TimeTracker(nb_epoch=params["nb_epoch"])

        tt.train_start()
        model.fit(train_loader, valid_loader)
        tt.train_end()

        tt.test_start()
        anomaly_score, anomaly_label = model.predict_prob(
            test_loader, test_windows_label
        )
        tt.test_end()

        train_anomaly_score, anomaly_label = model.predict_prob(
            train_loader, test_windows_label
        )

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
            experiment_id.rsplit('_', 1)[0]  # 使用实验ID中最后一个下划线之前的部分作为模型名称
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
