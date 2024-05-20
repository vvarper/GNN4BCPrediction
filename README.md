# Uveiling Agents' Confidence in Opinion Dynamics Models via Graph Neural Networks

Supplementary material with the code and datasets used in the paper
***Unveiling
Agents' Confidence in Opinion Dynamics Models via Graph Neural Network***,
co-authored by Víctor A. Vargas-Pérez, Jesús Giráldez-Cru, Pablo Mesejo, and
Oscar Cordón.

## Experiments pipeline

The folder `scripts` contains the code used to run each step of the
experimentation. The pipeline is divided into the following steps:

1. `create_synthetic_topologies.py` : creates the 9 synthetic topologies and
   store them in the `data/topologies/synthetic` folder in GML format.
2. `process_real_topologies.py` : download the real-world topologies, process
   them (get the largest connected component) and store them in
   the `data/topologies/real` folder in GML format.'
3. `create_datasets.py` : creates the dataset associate with each topology. The
   datasets are stored in the `data/datasets` folder as a Pytorch tensor. In
   addition, the scripts stores each datasaet as a JSON file in
   the `data/graphs` before splitting the datasets into training, validation,
   and test sets.
4. `train_model` : parametrized script to train a model on a given synthetic
   dataset with a specific hyperparameter configuration (learning rate, number
   of layers L, number of hidden units H, and batch size). The script stores
   the model in the `models/tuning/` folder, as well as the corresponding loss
   curve in the `data/tuning_results` folder.
5. `test_hyperparameter-tuning` : script to test the hyperparameter tuning
   procedure. The script generates a `results.csv` file in
   the `data/tuning_results` folder for each threshold scenario ('hom', 'com)
   and layer type ('mlp', 'gcn', 'sage', 'gatv2') with the MSE result in
   training and validation of each hyperparameter configuration. In addition,
   it copies the best model for each scenario/layer to the `models/best/`
   folder.
6. `test_model` : script to test the best model for each scenario/layer in
   the synthetic test dataset and the real-world datasets. The script stores
   the results in the `data/test_results/`, which include CSV files with the
   MSE, MAE, MAPE and R2 metrics, plots with the predicted vs. true values, and
   the confidence values, and CSV files with statistical tests results.

## Supplementary material and scripts

The folder `scripts` includes two additional scripts to generate figures and
tables to describe the characteristics of the topologies and the datasets used:

1. `describe_topologies` : script to generate figures with the degree distribution of each topology, as well as a CSV file with the characteristics of all of them.
2. `simulate_hk_graphs` : script to simulate the HK model in every graph and generate the corresponding plots with the evolution of the opinion distribution. Thus, we can see if every graph reach a stationary state.

## License

Read [LICENSE](./LICENSE).