import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import src.constants as constants

def load_filtered_data():
	final_path = os.path.join(constants.INTERMEDIATE_DIR, "final_dataset.pkl")
	df = pd.read_pickle(final_path)
	print(f"Original dataset shape: {df.shape}")
	df = df[df['age'] >= 18].copy()
	print(f"Filtered to adults (age >= 18). New shape: {df.shape}")
	print("Numeric summary of adult subset:")
	print(df.describe())
	return df

def plot_age_distribution(df):
	plt.figure(num="Age Distribution", figsize=(8, 6))
	df['age'].hist(bins=20)
	plt.title("Age Distribution for Patients 18+")
	plt.xlabel("Age")
	plt.ylabel("Frequency")
	path = os.path.join(constants.CHARTS_DIR, "age_distribution_adults.png")
	plt.savefig(path)
	print(f"Saved age distribution chart to {path}")

def plot_gender_distribution(df):
	plt.figure(num="Gender Distribution", figsize=(6, 4))
	df['GENDER'].value_counts().plot(kind='bar')
	plt.title("Gender Distribution (Adults)")
	plt.xlabel("Gender")
	plt.ylabel("Count")
	path = os.path.join(constants.CHARTS_DIR, "gender_distribution_adults.png")
	plt.savefig(path)
	print(f"Saved gender distribution chart to {path}")

def plot_label_distribution(df):
	plt.figure(num="Sepsis Label Distribution", figsize=(6, 4))
	df['label'].value_counts().plot(kind='bar')
	plt.title("Sepsis Label Distribution (Adults)")
	plt.xlabel("Label (0: No Sepsis, 1: Sepsis)")
	plt.ylabel("Count")
	path = os.path.join(constants.CHARTS_DIR, "label_distribution_adults.png")
	plt.savefig(path)
	print(f"Saved label distribution chart to {path}")

def plot_time_window_distribution(df):
	if 'time_window' in df.columns:
		plt.figure(num="Time Window Distribution", figsize=(6, 4))
		df['time_window'].hist(
			bins=range(int(df['time_window'].min()), int(df['time_window'].max()) + 2)
		)
		plt.title("Time Window Distribution (Adults)")
		plt.xlabel("Time Window")
		plt.ylabel("Frequency")
		path = os.path.join(constants.CHARTS_DIR, "time_window_distribution_adults.png")
		plt.savefig(path)
		print(f"Saved time window distribution chart to {path}")

def plot_correlation_matrix(df):
	drop_cols = ['SUBJECT_ID', 'HADM_ID', 'time_window']
	df = df.drop(columns=[col for col in drop_cols if col in df.columns])

	numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
	corr = df[numeric_cols].corr()

	sns.set(font_scale=0.7)
	plt.figure(num="Correlation Matrix", figsize=(12, 10))
	sns.heatmap(corr, cmap='coolwarm', annot=False, square=False)
	plt.title("Correlation Matrix (Adults)")
	plt.xticks(rotation=45, ha='right')
	plt.yticks(rotation=0)
	plt.tight_layout()

	path = os.path.join(constants.CHARTS_DIR, "correlation_matrix_adults.png")
	plt.savefig(path)
	print(f"Saved correlation matrix chart to {path}")

def main():
	df = load_filtered_data()
	plot_age_distribution(df)
	plot_gender_distribution(df)
	plot_label_distribution(df)
	plot_time_window_distribution(df)
	plot_correlation_matrix(df)
	plt.show()

if __name__ == "__main__":
	os.makedirs(constants.CHARTS_DIR, exist_ok=True)
	main()
