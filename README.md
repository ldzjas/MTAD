### 虚拟环境安装

> 见./安装命令.txt

### 运行不同baseline

示例
```
cd benchmark
python lstm_benchmark.py
```

### Datasets
通过./benchmark/benchmark_config/dataset_config/SWAT.yaml配置，放在./data/SWAT下

### Models integrated in this tool

**General Machine Learning-based Models**

| Model   | Paper reference                                              |
| :------ | :----------------------------------------------------------- |
| PCA     | **[2003]** Shyu M L, Chen S C, Sarinnapakorn K, et al. A novel anomaly detection scheme based on principal component classifier |
| iForest | **[ICDM'2008]** Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou: Isolation Forest |
| LODA    | **[Machine Learning'2016]** Tomás Pevný. Loda**:** Lightweight online detector of anomalies |

**Deep Learning-based Models**

| Model       | Paper reference                                              |
| :---------- | :----------------------------------------------------------- |
| AE          | **[AAAI'2019]** Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca Benini. Anomaly Detection Using Autoencoders in High Performance Computing Systems |
| LSTM        | **[KDD'2018]** Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, Tom Söderström. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding |
| LSTM-VAE    | **[Arxiv'2017]** A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder |
| DAGMM       | **[ICLR'2018]** Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Dae-ki Cho, Haifeng Chen. Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection |
| MSCRED      | **[AAAI'19]** Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, Nitesh V. Chawla. A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data. |
| OmniAnomaly | **[KDD'2019]** Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei. Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network |
| MTAD-GAT | **[ICDM'2020]** Multivariate Time-series Anomaly Detection via Graph Attention Networks |
| USAD | **[KDD'2020]** USAD: UnSupervised Anomaly Detection on Multivariate Time Series. |
| InterFusion | **[KDD'2021]** Zhihan Li, Youjian Zhao, Jiaqi Han, Ya Su, Rui Jiao, Xidao Wen, Dan Pei. Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding |
| TranAD | **[VLDB'2021]** TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data |
| RANSynCoders | **[KDD'2021]** Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization |
| AnomalyTransformer | **[ICLR'2022]** Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy |
| GANF | **[ICLR'2022]** Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series |
