"""Code fot the deep learning model."""

import logging
import torch.nn as nn
import lightning as L

import src.constants as CONST
import src.model.configuration as CONFIG

logger = logging.getLogger(__name__)


class DeepLearningModel(L.LightningModule):
    """Deep learning model for river bank erosion prediction."""

    def __init__(self, configuration: CONFIG.ModelConfiguration):
        """Initialise the model with the configuration."""
        super().__init__()
        self.configuration = configuration
