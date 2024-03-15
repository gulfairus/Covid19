import os

import pytest

from taxifare.params import *
from taxifare.interface.workflow import *
from tests.test_base import TestBase


class TestPrefect(TestBase):

    def test_prefect_flow_name_is_not_null(self):
        """
        verify that the prefect parameters are correctly set
        """
        flow_name = PREFECT_FLOW_NAME
        assert flow_name is not None, 'PREFECT_FLOW_NAME variable is not defined'

    def test_prefect_log_level_is_warning(self):
        """
        verify that the prefect parameters are correctly set
        """
        log_level = PREFECT_LOG_LEVEL
        assert log_level == 'WARNING'

    def test_prefect_tasks(self):
        """
        verify that the all the prefect tasks are created
        """
        assert preprocess_new_data.name == 'preprocess_new_data', "preprocess_new_data task is not defined"
        assert evaluate_production_model.name == 'evaluate_production_model', "evaluate_production_model task is not defined"
        assert re_train.name == 're_train', "re_train task is not defined"
        assert transition_model.name == 'transition_model', "transition_model task is not defined"
        assert notify.name == 'notify', "notify task is not defined"

    def test_prefect_flow(self):
        """
        test that there is a prefect flow
        """
        assert train_flow.version is not None, "train_flow is not turned into a flow"
