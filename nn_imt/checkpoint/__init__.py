import logging
import os
import copy
import shutil

from machine_translation.checkpoint import RunExternalValidation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IMTRunExternalValidation(RunExternalValidation):
    """
    Run external validation for IMT models
    """
    def build_evaluation_command(self, config_filename):
        """Note that nn_imt must be available as a module for this command to work"""
        command = ['python', '-m', 'nn_imt', '-m', 'evaluate', config_filename]
        return command

