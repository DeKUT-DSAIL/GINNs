# RUL Prediction in LFP Batteries: Comparison of Gompertz, LSTM, and Gompertz-Informed LSTM Models for Interpreta-bility and Accuracy 

## Description
This repository contains the codebase and analysis for predicting the Remaining Useful Life (RUL) and State of Health (SoH) of Lithium Iron Phosphate (LFP) batteries. The project evaluates the trade-off between the predictive accuracy of purely data-driven black-box models (LSTMs) and the physical interpretability of empirical models (the Gompertz function). It introduces hybrid gray-box architectures known as Gompertz-Informed LSTMs (GILSTMs) designed to constrain predictions to follow physical laws of battery degradation.

The code and analysis in this repository support the findings presented in the paper:
*"RUL Prediction in LFP Batteries: Comparison of Gompertz, LSTM, and Gompertz-Informed LSTM Models for Interpretability and Accuracy"* submitted to *Batteries by MDPI*.

## Repository Structure
Below is the organizational structure of the codebase [generated using project tree](https://project-tree-generator.netlify.app/generate-tree):

```text
gi-lstms/
├── .gitattributes
├── .gitignore
├── current-plots/
│   └── 1-1-current-plot.png
├── data/
│   └── external/
│       └── HUST/
│           └── 1-1.pkl.zip            # Raw HUST battery dataset files
├── LICENSE.txt                        # Usage rights
├── main/                              # Core execution files and saved results
│   ├── GILSTM_1_rmse_results.pkl
│   ├── GILSTM_2_rmse_results.pkl
│   ├── GILSTM_3_rmse_results.pkl
│   ├── gompertz_results.pkl
│   ├── gompertz_rmse_results.pkl
│   ├── hust_rmse_results.ipynb
│   ├── hust-ginn-data-pipeline.ipynb      # Main data processing pipeline
│   ├── hust-ginn-modelling-pipeline.ipynb # Main model training/testing pipeline
│   ├── models/                            # Saved model weights (.pth)
│   │   ├── best_lstm_model-window-100_model_pinn_data_all_cycle_to_cycle.pth
│   │   ├── best_moe_model_window_100.pth
│   │   ├── SoH-to-kab-best_lstm_model-window-100_model_pinn_data_all.pth
│   │   ├── SoH-to-kab-best_lstm_model-window-100_model_pinn_data_high.pth
│   │   ├── SoH-to-kab-best_lstm_model-window-100_model_pinn_data_low.pth
│   │   ├── SoH-to-kab-best_lstm_model-window-100_model_pinn_data_mid.pth
│   │   ├── SoH-to-RUL-best_lstm_model-window-100.pth
│   │   ├── SoH-to-SoH-best_lstm_model-window-100.pth
│   │   ├── SoH-to-SoH-best_lstm_model-window-100v0.pth
│   │   └── SoH-to-SoH-best_lstm_model-window-100v1.pth
│   ├── SOH_to_RUL_rmse_results.pkl
│   ├── SOH_to_SoH_RUL_rmse_results.pkl
│   └── unified_gompertz.pkl
├── notebooks/                         # Experimental & modeling notebooks
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── ablation-study.ipynb           # Ablation study analysis
│   ├── battery-operationability.ipynb # Operationalization insights
│   ├── hust-ginn-data-pipeline.ipynb
│   ├── hust-ginn-modelling-pipeline.ipynb
│   ├── train-nb-4-pytorch-soh-to-kab-parameters-lstm_cycle_to_cycle.ipynb
│   ├── train-nb-4-pytorch-soh-to-kab-parameters-lstm-2-losses.ipynb
│   ├── train-nb-4-pytorch-soh-to-kab-parameters-lstm.ipynb
│   ├── train-nb-4-pytorch-soh-to-rul-lstm.ipynb
│   └── train-nb-4-pytorch-soh-to-soh-lstm.ipynb
├── raw-code/                          # Initial data extraction scripts
│   ├── huazhong-ust-li-battery-data-extraction.ipynb
│   ├── hust-battery-data-acquisition.ipynb
│   └── plot-gompertz-parameters.ipynb
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── results/                           # Generated plots and CSV summaries
│   ├── capacity-fade-cno-plot/
│   │   └── 1-1.png
│   ├── capacity-fade-time-plot/
│   │   └── 1-1.png
│   ├── csv-summary/
│   │   └── 1-1.csv
│   ├── csvs/
│   │   └── 1-1.csv
│   ├── current-plots/
│   │   └── 1-1-current-plot.png
│   ├── plots/
│   │   ├── 1-1-first-10-plot.png
│   │   └── 1-1-full-plot.png
│   └── SoH-cno-plot/
│       └── 1-1.png
└── src/                               # Source code modules
    ├── __init__.py
    ├── data_loader.py
    ├── models.py
    ├── SoC-calculation-HUST.py
    ├── SoC-calculation-MATR.py
    ├── SoH-calculation-HUST.py
    ├── SoH-calculation-MATR.py
    └── train.py
```

## Requirements
To run the notebooks and scripts in this repository, you will need Python installed along with the packages specified in the requirements file. Core dependencies include frameworks for deep learning and data manipulation:

* `torch` (PyTorch)
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

To install all required dependencies, run:
```bash
pip install -r requirements.txt
```

## Installation and Usage

Follow these steps to set up the environment and run the core pipelines:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DeKUT-DSAIL/gi-lstms.git
   cd gi-lstms
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the core pipelines:**
   The primary workflow is divided into two main Jupyter Notebooks located in the `main/` directory. Open and run them sequentially:
   * **Data Preparation:** Open `main/hust-ginn-data-pipeline.ipynb` to process the raw HUST data, calculate SoC/SoH, and extract the Gompertz parameters.
   * **Modeling & Inference:** Open `main/hust-ginn-modelling-pipeline.ipynb` to train the LSTMs and GILSTMs, and to evaluate their performance on RUL prediction.

## Data Access

* **Raw Battery Data:** The original Huazhong University of Science and Technology (HUST) battery dataset used in this study is openly available on Mendeley Data at: [https://data.mendeley.com/datasets/nsc7hnsg4s/2](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
* **Gompertz Parameters Dataset:** The generated derivative data, including calculated Gompertz parameters (k, a, b), is available within this repository.
* **Further Access:** If you require specific processed data partitions not included in the repository, please contact `yuri.njathi@dkut.ac.ke`.

## Citation

If you use this repository, please cite:
```bibtex
@article{yourcitation2025,
  author = {Author Name, Co-Author Name},
  title = {Your Paper Title},
  journal = {},
  year = {2025},
  volume = {X},
  pages = {XX-XX},
  doi = {XX.XXXXX/journal.xxxxxx}
}
```

## Acknowledgements
This work was conducted as part of the Artificial Intelligence for Development (AI4D) program, with the financial support of the UK government’s Foreign, Commonwealth, and Development Office (FCDO) and Canada’s International Development Research Centre (IDRC). 

We appreciate the support from Arm and Google.org to the Centre for Data Science and Artificial Intelligence (DSAIL). This work was also supported by a grant from the Swiss National Supercomputing Centre (CSCS) under project ID g164 on Alps. Additional computational resources for modeling and benchmarking results were obtained through Kaggle.
