# Airport Traffic Forecasting – Graph-Augmented Machine Learning

This repository contains the complete code used in the study "Graph-augmented temporal forecasting framework for airport traffic prediction". The project includes all scripts required for preprocessing the dataset, generating graph embeddings, training the proposed graph-augmented machine learning model, and reproducing the main results reported in the manuscript. The dataset is publicly available on Kaggle at: https://www.kaggle.com/datasets/mohammedalsubaie/king-khalid-international-airport-flights-dataset. Download the dataset manually and place the file `flights_RUH.parquet` inside a folder named `data/` in the root directory of the project. The `data/` folder is not included in this repository and must be created manually.

To reproduce the main results, follow these steps after placing the dataset in the correct folder:  
1. Run `python src/run_pipeline.py` to preprocess the dataset and generate hourly features.  
2. Run `python src/03_graph_embedding.py` to construct the graph and generate Node2Vec embeddings.  
3. Run `python src/04_train_xgboost_with_graph.py` to train the proposed graph-augmented XGBoost model.  
4. Run `python src/04b_train_classic_ml.py` to train classical baseline models for comparison.  

All output files, including generated features, embeddings, predictions, metrics, and figures, will be saved automatically inside an `outputs/` folder.

Install dependencies using:  
`pip install -r requirements.txt`  
The codebase was tested on Python 3.9+.


Please cite the associated research article once it is published.
