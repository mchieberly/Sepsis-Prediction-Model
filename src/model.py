import torch.nn as nn
import src.constants as constants

class SepsisPredictor(nn.Module):
	"""MLP with four hidden layers and dropout."""
	def __init__(self, input_dim):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, constants.MODEL_HIDDEN)
		self.bn1 = nn.BatchNorm1d(constants.MODEL_HIDDEN)
		self.fc2 = nn.Linear(constants.MODEL_HIDDEN, constants.MODEL_HIDDEN)
		self.bn2 = nn.BatchNorm1d(constants.MODEL_HIDDEN)
		self.out = nn.Linear(constants.MODEL_HIDDEN, constants.OUTPUT_DIM)
		self.dropout = nn.Dropout(constants.MODEL_DROPOUT)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.dropout(self.relu(self.bn1(self.fc1(x))))
		x = self.dropout(self.relu(self.bn2(self.fc2(x))))
		return self.out(x)
