from pathlib import Path
import yaml

def load_config():
    with open("configs/configs.yaml", "r") as f:
        return yaml.safe_load(f)
    
configs=load_config()

base_path=Path(__file__).resolve().parent.parent
data_path=base_path/configs["path"]["data_dir"]
training_path=data_path/configs["path"]["training_path"]

SEED=configs["experiment_values"]["seed"]
TEST_SIZE=configs["experiment_values"]["test_size"]
SCORING=configs["experiment_values"]["scoring"]
N_JOBS=configs["experiment_values"]["n_jobs"]
CV=configs["experiment_values"]["cv"]
DATASET_NAME=configs["dataset_name"]["name"]

N_SPLITS=configs["stratified_fold"]["n_splits"]
SHUFFLE=configs["stratified_fold"]["shuffle"]

LOGISTIC_C=configs["model_parameters"]["logistic_parameters"]["C"]

DECI_TREE_CRITIRION=configs["model_parameters"]["decision_tree_parameters"]["criterion"]
DECI_TREE_MAX_DEPTH=configs["model_parameters"]["decision_tree_parameters"]["max_depth"]
DECI_TREE_MIN_SAMPLES_LEAF=configs["model_parameters"]["decision_tree_parameters"]["min_samples_leaf"]

RANDOM_FOREST_CRITIRION=configs["model_parameters"]["random_forest_parameters"]["criterion"]
RANDOM_FOREST_N_ESTIMATORS=configs["model_parameters"]["random_forest_parameters"]["n_estimators"]
RANDOM_FOREST_MAX_DEPTH=configs["model_parameters"]["random_forest_parameters"]["max_depth"]

XGBOOST_LEARNING_RATE=configs["model_parameters"]["xgboost_parameters"]["learning_rate"]
XGBOOST_N_ESTIMATORS=configs["model_parameters"]["xgboost_parameters"]["n_estimators"]
XGBOOST_MAX_DEPTH=configs["model_parameters"]["xgboost_parameters"]["max_depth"]

scoring=list(configs["scoring_parameters"]["metrics"].values())

class model_parameters:
    
    logistic_parameters={"C":LOGISTIC_C
                         }
    
    decision_tree_parameters={"criterion":DECI_TREE_CRITIRION,
                              "max_depth":DECI_TREE_MAX_DEPTH,
                              "min_samples_leaf":DECI_TREE_MIN_SAMPLES_LEAF
                              }
    
    random_forest_parameters={"n_estimators":RANDOM_FOREST_N_ESTIMATORS,
                              "max_depth":RANDOM_FOREST_MAX_DEPTH,
                              "criterion":RANDOM_FOREST_CRITIRION
                              }
    
    xgboost_parameters={"n_estimators":XGBOOST_N_ESTIMATORS,
                        "max_depth":XGBOOST_MAX_DEPTH,
                        "learning_rate":XGBOOST_LEARNING_RATE
                        }