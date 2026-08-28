# FL-Benchmarking

Comparing federated, centralised, and local learning across multisite clinical prediction tasks

CHOIR_ModelOps/
│
├── config/
│   ├── model_config.yaml
│   ├── site_config.yaml
│   └── logging_config.yaml
│
├── data_processing/
│   ├── extract_data.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   └── validate_data.py
│
├── model/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── model_utils.py
│
├── federated_learning/
│   ├── client.py
│   ├── server.py
│   ├── aggregation.py
│   └── communication.py
│
├── deployment/
│   ├── deploy_model.py
│   ├── monitor_model.py
│   ├── rollback.py
│   └── version_control.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_federated.py
│
├── scripts/
│   ├── run_training.sh
│   ├── run_validation.sh
│   └── run_pipeline.sh
│
├── requirements.txt
├── README.md
└── main.py
