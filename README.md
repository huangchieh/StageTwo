# Retrain the structure discovery model

StageTwo is the step to 
1. retrain the structure discovery machine learning model using different datasets. 
2. get the structure predictions by feeding the real AFM images to the trained model. 

The performance evaluation for these predictions are in [this repository](https://github.com/huangchieh/StructureMetrics). 

Workflow: 
1. Install [ml-spm](https://github.com/SINGROUP/ml-spm/tree/main) 
2. Train posnet and graphnet using original dataset, style translated dataset. 
3. Test trained models using experimental dataset.

