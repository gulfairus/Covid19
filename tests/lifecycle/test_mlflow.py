import os
import re

import pandas as pd
import pytest
import mlflow

from detection.params import *
from tests.test_base import TestBase

class TestMlflow(TestBase):
    def test_model_target_is_mlflow(self):
        """
        verify that the mlflow parameters are correctly set
        """
        model_target = MODEL_TARGET

        assert model_target == 'mlflow', 'Check the value of MODEL_TARGET'

    def test_mlflow_experiment_is_not_null(self):
        """
        verify that the mlflow parameters are correctly set
        """
        experiment = MLFLOW_EXPERIMENT

        assert experiment is not None, 'Please fill in the MLFLOW_EXPERIMENT variable'

    def test_mlflow_model_name_is_not_null(self):
        """
        verify that the mlflow parameters are correctly set
        """
        model_name = MLFLOW_MODEL_NAME

        assert model_name is not None, 'Please fill in the MLFLOW_MODEL_NAME variable'

    def test_mlflow_experiment_exists(self):
        """
        verify that the mlflow experiment exists
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow_client = mlflow.tracking.MlflowClient()


        experiment_id = mlflow_client.get_experiment_by_name(MLFLOW_EXPERIMENT).experiment_id

        assert experiment_id is not None, f'Please create the experiment {MLFLOW_EXPERIMENT} in mlflow by doing your first run'

    def test_mlflow_model_exists(self):
        """
        verify that the mlflow model exists
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow_client = mlflow.tracking.MlflowClient()

        model = mlflow_client.get_registered_model(MLFLOW_MODEL_NAME)

        assert model is not None, f'Please create the model {MLFLOW_MODEL_NAME} in mlflow'

    def test_mlflow_model_in_production(self):
        """
        Verify that a version of the model is in production
        """
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        mlflow_client = mlflow.tracking.MlflowClient()

        production_model = mlflow_client.get_latest_versions(MLFLOW_MODEL_NAME, stages=['Production'])

        assert len(production_model) > 0, f'Please create a version of the model {MLFLOW_MODEL_NAME} in production'
