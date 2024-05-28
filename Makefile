.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y detection || :
	@pip install -e .

run_preprocess_svm:
	python -c 'from detection.svm.interface.main import preprocess; preprocess()'

run_train_svm:
	python -c 'from detection.svm.interface.main import train; train()'

run_train_xgboost:
	python -c 'from detection.xgboost.interface.main import train; train()'

run_train_cnn_scratch:
	python -c 'from detection.cnn_scratch.interface.main import train; train()'

run_train_cnn_pre:
	python -c 'from detection.cnn_pre.interface.main import train; train()'

run_train_transformer:
	python -c 'from detection.transformer.interface.main import train; train()'

run_pred:
	python -c 'from taxifare.interface.main import pred; pred()'

run_evaluate:
	python -c 'from taxifare.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

##################### TESTS #####################
test_gcp_setup:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_wagon_project

default:
	cat tests/cloud_training/test_output.txt

test_kitt:
	@echo "\n 🧪 computing and saving your progress at 'tests/cloud_training/test_output.txt'...(this can take a while)"
	@pytest tests/cloud_training -c "./tests/pytest_kitt.ini" 2>&1 > tests/cloud_training/test_output.txt || true
	@echo "\n 🙏 Please: \n git add tests \n git commit -m 'checkpoint' \n ggpush"


test_mlflow_config:
	@pytest \
	tests/lifecycle/test_mlflow.py::TestMlflow::test_model_target_is_mlflow \
	tests/lifecycle/test_mlflow.py::TestMlflow::test_mlflow_experiment_is_not_null \
	tests/lifecycle/test_mlflow.py::TestMlflow::test_mlflow_model_name_is_not_null

test_prefect_config:
	@pytest \
	tests/lifecycle/test_prefect.py::TestPrefect::test_prefect_flow_name_is_not_null \
	tests/lifecycle/test_prefect.py::TestPrefect::test_prefect_log_level_is_warning

test_preprocess:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_preprocess

test_train:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_train

test_evaluate:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_evaluate

test_pred:
	@pytest tests/cloud_training/test_main.py::TestMain::test_route_pred

test_main_all: test_preprocess test_train test_evaluate test_pred

test_gcp_project:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_project_id

test_gcp_bucket:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_bucket_name

test_big_query:
	@pytest \
	tests/cloud_training/test_cloud_data.py::TestCloudData::test_big_query_dataset_variable_exists \
	tests/cloud_training/test_cloud_data.py::TestCloudData::test_cloud_data_create_dataset \
	tests/cloud_training/test_cloud_data.py::TestCloudData::test_cloud_data_create_table

test_vm:
	tests/cloud_training/test_vm.py


################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR=~/.lewagon/covid19
#HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/
#GS_DIR=gs://datascience-mlops/taxi-fare-ny

show_sources_all:
	-ls -laR ~/.lewagon/covid19/*
	-gsutil ls gs://${BUCKET_NAME}/models

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/covid19/data/
	mkdir ~/.lewagon/covid19/data/raw
	mkdir ~/.lewagon/covid19/data/processed
	mkdir ~/.lewagon/covid19/training_outputs
	mkdir ~/.lewagon/covid19/training_outputs/metrics
	mkdir ~/.lewagon/covid19/training_outputs/models
	mkdir ~/.lewagon/covid19/training_outputs/params
	mkdir ~/.lewagon/covid19/training_outputs/metrics/cnn_pre
	mkdir ~/.lewagon/covid19/training_outputs/models/cnn_pre
	mkdir ~/.lewagon/covid19/training_outputs/params/cnn_pre
	mkdir ~/.lewagon/covid19/training_outputs/metrics/cnn_scratch
	mkdir ~/.lewagon/covid19/training_outputs/models/cnn_scratch
	mkdir ~/.lewagon/covid19/training_outputs/params/cnn_scratch
	mkdir ~/.lewagon/covid19/training_outputs/metrics/transformer
	mkdir ~/.lewagon/covid19/training_outputs/models/transformer
	mkdir ~/.lewagon/covid19/training_outputs/params/transformer
	mkdir ~/.lewagon/covid19/training_outputs/metrics/xgboost
	mkdir ~/.lewagon/covid19/training_outputs/models/xgboost
	mkdir ~/.lewagon/covid19/training_outputs/params/xgboost
	mkdir ~/.lewagon/covid19/training_outputs/metrics/svm
	mkdir ~/.lewagon/covid19/training_outputs/models/svm
	mkdir ~/.lewagon/covid19/training_outputs/params/svm


#reset_gcs_files:
#	-gsutil rm -r gs://${BUCKET_NAME}
#	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files
