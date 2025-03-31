import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import src.constants as constants
from src.model import SepsisPredictor
from src.utils import save_model

class MIMICDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.tensor(X.values, dtype=torch.float32)
		self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

def load_data():
	data = pd.read_pickle(os.path.join(constants.INTERMEDIATE_DIR, "final_dataset.pkl"))
	X = data.drop(columns=["SUBJECT_ID", "HADM_ID", "time_window", "INTIME", "label"])
	y = data["label"]
	X["GENDER"] = X["GENDER"].map({"M": 0, "F": 1}).fillna(constants.DEFAULT_GENDER)
	print(f"Loaded {len(data)} samples. Class distribution: {y.value_counts().to_dict()}")
	X_train, X_temp, y_train, y_temp = train_test_split(
		X, y, test_size=constants.SPLIT_TEST_SIZE, stratify=y, random_state=constants.RANDOM_STATE)
	X_val, X_test, y_val, y_test = train_test_split(
		X_temp, y_temp, test_size=constants.SPLIT_VAL_RATIO, stratify=y_temp, random_state=constants.RANDOM_STATE)
	print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
	return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val, X_test):
	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
	X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
	return X_train, X_val, X_test

def get_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test):
	train_dataset = MIMICDataset(X_train, y_train)
	val_dataset = MIMICDataset(X_val, y_val)
	test_dataset = MIMICDataset(X_test, y_test)
	train_loader = DataLoader(train_dataset, batch_size=constants.TRAIN_BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=constants.TRAIN_BATCH_SIZE)
	test_loader = DataLoader(test_dataset, batch_size=constants.TRAIN_BATCH_SIZE)
	print(f"Batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
	return train_loader, val_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, device):
	model.train()
	total_loss = 0
	for X, y in loader:
		X, y = X.to(device), y.to(device)
		optimizer.zero_grad()
		loss = criterion(model(X), y)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * X.size(0)
	return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, threshold):
	model.eval()
	total_loss, y_true, y_pred = 0, [], []
	with torch.no_grad():
		for X, y in loader:
			X, y = X.to(device), y.to(device)
			outputs = model(X)
			total_loss += criterion(outputs, y).item() * X.size(0)
			preds = (torch.sigmoid(outputs) >= threshold).float()
			y_true.extend(y.cpu().numpy())
			y_pred.extend(preds.cpu().numpy())
	loss = total_loss / len(loader.dataset)
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	acc = (y_true == y_pred).mean()
	return loss, acc, y_true, y_pred

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	X_train, X_val, X_test, y_train, y_val, y_test = load_data()
	X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
	train_loader, val_loader, test_loader = get_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test)

	model = SepsisPredictor(X_train.shape[1]).to(device)
	pos_weight = torch.tensor([len(y_train[y_train == 0]) / len(y_train[y_train == 1])],
							  dtype=torch.float32).to(device)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	optimizer = optim.Adam(model.parameters(), lr=constants.TRAIN_LEARNING_RATE,
						   weight_decay=constants.TRAIN_WEIGHT_DECAY)

	# Initial evaluation
	val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, threshold=constants.PREDICTION_THRESHOLD)
	print(f"Initial validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

	best_val_loss = float('inf')
	patience_counter = 0
	best_model_state = None
	start_time = time.time()

	for epoch in range(constants.TRAIN_NUM_EPOCHS):
		train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, threshold=constants.PREDICTION_THRESHOLD)
		print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model_state = model.state_dict()

	total_time = time.time() - start_time
	print(f"Training complete in {total_time:.2f} seconds")
	model.load_state_dict(best_model_state)

	# Final evaluation
	test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, threshold=constants.PREDICTION_THRESHOLD)
	cm = confusion_matrix(y_true, y_pred)
	precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
	recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
	f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

	print("\nFinal Evaluation:")
	print(f"Loss={test_loss:.4f}, Acc={test_acc:.4f}")
	print(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
	print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

	save_model(model, os.path.join(constants.INTERMEDIATE_DIR, "model.pt"))

if __name__ == "__main__":
	main()
