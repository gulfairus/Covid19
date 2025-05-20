.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y classification || :
	@pip install -e .

run_preprocess_images_001:
	python -c 'from classification.preprocess.densenet121 import extract_features_001; extract_features_001()'

run_preprocess_svm:
	python -c 'from classification.svm.interface.main import preprocess; preprocess()'

run_preprocess_resnet50:
	python -c 'from classification.resnet50.ml_logic.main import preprocess; preprocess()'

run_train_svm:
	python -c 'from classification.svm.interface.main import train; train()'

run_train_xgboost:
	python -c 'from classification.xgboost.interface.main import train; train()'

run_train_cnn_scratch:
	python -c 'from classification.cnn_scratch.interface.main import train; train()'

run_train_resnet50:
	python -c 'from classification.resnet50.interface.main import train; train()'

run_train_transformer:
	python -c 'from classification.transformer.interface.main import train; train()'

run_pred:
	python -c 'from taxifare.interface.main import pred; pred()'

run_evaluate:
	python -c 'from taxifare.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate


################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR=~/.database/lung_cancer

show_sources_all:
	-ls -laR ~/.database/lung_cancer/*
	-gsutil ls gs://${BUCKET_NAME}/models

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.database/lung_cancer/data/
	mkdir ~/.database/lung_cancer/data/raw
	mkdir ~/.database/lung_cancer/data/processed
	mkdir ~/.database/lung_cancer/data/processed/features
	mkdir ~/.database/lung_cancer/data/processed/features/densenet121
	mkdir ~/.database/lung_cancer/training_outputs
	mkdir ~/.database/lung_cancer/training_outputs/metrics
	mkdir ~/.database/lung_cancer/training_outputs/models
	mkdir ~/.database/lung_cancer/training_outputs/params
	mkdir ~/.database/lung_cancer/training_outputs/metrics/resnet50
	mkdir ~/.database/lung_cancer/training_outputs/models/resnet50
	mkdir ~/.database/lung_cancer/training_outputs/params/resnet50


#reset_gcs_files:
#	-gsutil rm -r gs://${BUCKET_NAME}
#	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files
