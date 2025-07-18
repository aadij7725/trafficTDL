# Combinatorial Complex Neural Network (CCNN) for Traffic Forecasting

![Combinatorial Complex Neural Network Model Architecture](./model_architecture.png "Model Architecture")

This repository contains code for traffic forecasting using **Combinatorial Complex Neural Networks (CCNN)**, a topological deep learning approach that leverages combinatorial structures for spatiotemporal prediction. 

## Requirements

- Python 3.7+
- All dependencies are listed in `requirements.txt`

Install requirements with:

```bash
pip install -r requirements.txt
```

## Data Preparation

Traffic datasets should be placed in the following directories:

```
data/
    METR-LA/
        metr-la.h5
        nodelocations.pkl
    PEMS-BAY/
        pems-bay.h5
        nodelocations.pkl
```

> **Note:**  
> The `nodelocations.pkl` files used in this repository was obtained from a third party:  https://github.com/RiccardoSpolaor/Verbal-Explanations-of-Spatio-Temporal-Graph-Neural-Networks-for-Traffic-Forecasting

To prepare data for training and evaluation, run:

```bash
python prepare_data.py
```

## Model Architecture

> ![Combinatorial Complex Neural Network Model Architecture](./model_architecture.png "Model Architecture")


## Pre-trained Models

Pre-trained CCNN models are available in this repository. All pre-trained models use the `.pt` file format.  
You can utilize these models directly for inference or further fine-tuning.

## Training

After preparing the data as described above, train the CCNN model with:

```bash
python train_traffic_ccnn.py
```

This script will automatically compute evaluation metrics during training.

## Evaluation

The following metrics are computed automatically by the training script:

- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**

## Applications

CCNN is a novel approach for spatiotemporal traffic forecasting and can be experimented with on any relevant traffic dataset. As this is a new model, its potential applications are broad and open for exploration.

## Citation

If you use CCNN or this codebase in your research, please cite this work (pending publication):

```
@article{jain2025tnn_traffic,
  title={TNNs: Topological Neural Networks for Traffic flow prediction},
  author={Ibrahem AlJabea, Aadi Jain, Mustafa Hajij},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025},
  note={Preprint, not yet published}
}
```

---

For any questions or contributions, please open an issue or pull request!
