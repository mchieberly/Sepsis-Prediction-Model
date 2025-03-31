import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import src.constants as constants
from src.utils import calc_age

pd.set_option('future.no_silent_downcasting', True)

def load_vitals():
	path = os.path.join(constants.RAW_DIR, "CHARTEVENTS.csv.gz")
	usecols = ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID", "VALUENUM"]
	valid_ids = [i for sub in constants.VITAL_SIGNS.values() for i in sub]
	chunks = []
	for chunk in tqdm(pd.read_csv(path, usecols=usecols, compression="gzip", chunksize=constants.CHUNKSIZE)):
		chunk = chunk[chunk["ITEMID"].isin(valid_ids)].dropna(subset=["VALUENUM"])
		chunk["CHARTTIME"] = pd.to_datetime(chunk["CHARTTIME"], errors="coerce")
		chunks.append(chunk)
	df = pd.concat(chunks, ignore_index=True)
	df.to_pickle(os.path.join(constants.INTERMEDIATE_DIR, "vitals.pkl"))
	print(f"Vitals loaded: {len(df)} rows.")

def load_labs():
	path = os.path.join(constants.RAW_DIR, "LABEVENTS.csv.gz")
	usecols = ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID", "VALUENUM"]
	valid_ids = [i for sub in constants.LAB_TESTS.values() for i in sub]
	df = pd.read_csv(path, usecols=usecols, compression="gzip")
	df = df[df["ITEMID"].isin(valid_ids)].dropna(subset=["VALUENUM"])
	df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], errors="coerce")
	df.to_pickle(os.path.join(constants.INTERMEDIATE_DIR, "labs.pkl"))
	print(f"Labs loaded: {len(df)} rows.")

def resample_features(df, icu_admissions):
	df = df.merge(icu_admissions[["SUBJECT_ID", "HADM_ID", "INTIME"]], on=["SUBJECT_ID", "HADM_ID"])
	df = df[(df["CHARTTIME"] >= df["INTIME"]) & 
			(df["CHARTTIME"] <= df["INTIME"] + pd.Timedelta(hours=constants.LABEL_WINDOW_HOURS))]
	df["time_window"] = ((df["CHARTTIME"] - df["INTIME"]).dt.total_seconds() // 
						 (constants.WINDOW_HOURS * constants.SECONDS_PER_HOUR)).astype(int)
	grouped = df.groupby(["SUBJECT_ID", "HADM_ID", "time_window", "ITEMID"])["VALUENUM"].agg(["mean", "min", "max"])
	pivoted = grouped.pivot_table(index=["SUBJECT_ID", "HADM_ID", "time_window"], columns="ITEMID")
	pivoted.columns = [f"{stat}_{item}" for stat, item in pivoted.columns]
	result = pivoted.reset_index()
	for col in result.columns:
		if col not in ["SUBJECT_ID", "HADM_ID", "time_window"]:
			result[f"diff_{col}"] = result.groupby(["SUBJECT_ID", "HADM_ID"])[col].diff().fillna(0)
	return result

def create_features(icu_admissions):
	vitals = pd.read_pickle(os.path.join(constants.INTERMEDIATE_DIR, "vitals.pkl"))
	labs = pd.read_pickle(os.path.join(constants.INTERMEDIATE_DIR, "labs.pkl"))
	vitals_window = resample_features(vitals, icu_admissions)
	labs_window = resample_features(labs, icu_admissions)
	merged = pd.merge(vitals_window, labs_window, on=["SUBJECT_ID", "HADM_ID", "time_window"], how="outer")
	merged.to_pickle(os.path.join(constants.INTERMEDIATE_DIR, "features.pkl"))
	print(f"Features created: {len(merged)} rows.")
	return merged

def load_sepsis_indicators(icu_admissions):
	vitals = pd.read_pickle(os.path.join(constants.INTERMEDIATE_DIR, "vitals.pkl"))
	labs = pd.read_pickle(os.path.join(constants.INTERMEDIATE_DIR, "labs.pkl"))
	combined = pd.concat([vitals, labs])
	combined = combined.merge(icu_admissions[["SUBJECT_ID", "HADM_ID", "INTIME"]], on=["SUBJECT_ID", "HADM_ID"])
	combined = combined[combined["CHARTTIME"] <= combined["INTIME"] + pd.Timedelta(hours=constants.LABEL_WINDOW_HOURS)]
	
	sepsis_times = []
	unique_groups = combined.groupby(["SUBJECT_ID", "HADM_ID"])
	for (subj, hadm), group in tqdm(unique_groups, total=len(unique_groups), desc="SIRS processing"):
		group = group.sort_values("CHARTTIME")
		sirs_counts = pd.DataFrame(index=group["CHARTTIME"].unique(), dtype=bool)
		for item, condition in constants.SIRS_CONDITIONS.items():
			item_data = group[group["ITEMID"] == item][["CHARTTIME", "VALUENUM"]]
			if not item_data.empty:
				item_data = item_data.groupby("CHARTTIME")["VALUENUM"].mean().reset_index()
				sirs_counts[f"sirs_{item}"] = item_data.set_index("CHARTTIME")["VALUENUM"].apply(condition).reindex(sirs_counts.index).fillna(False)
			else:
				sirs_counts[f"sirs_{item}"] = False
		sirs_cols = [f"sirs_{item}" for item in constants.SIRS_CONDITIONS]
		sirs_counts["sirs_count"] = sirs_counts[sirs_cols].sum(axis=1)
		sepsis_onsets = sirs_counts[sirs_counts["sirs_count"] >= 2].index
		if not sepsis_onsets.empty:
			sepsis_times.append(pd.DataFrame({
				"SUBJECT_ID": subj,
				"HADM_ID": hadm,
				"CHARTTIME": sepsis_onsets
			}))
	
	if sepsis_times:
		sepsis_times = pd.concat(sepsis_times, ignore_index=True)
	else:
		sepsis_times = pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "CHARTTIME"])
	return sepsis_times

def generate_labels(features, icu_admissions, sepsis_times):
	features = features.merge(icu_admissions[["SUBJECT_ID", "HADM_ID", "INTIME"]], on=["SUBJECT_ID", "HADM_ID"])
	def label_row(row):
		subj, hadm, window = row["SUBJECT_ID"], row["HADM_ID"], row["time_window"]
		icu_time = row["INTIME"]
		window_start = icu_time + pd.Timedelta(hours=window * constants.WINDOW_HOURS)
		window_end = window_start + pd.Timedelta(hours=constants.LABEL_WINDOW_HOURS)
		sepsis_events = sepsis_times[(sepsis_times["SUBJECT_ID"] == subj) & 
									 (sepsis_times["HADM_ID"] == hadm) &
									 (sepsis_times["CHARTTIME"] >= window_start) & 
									 (sepsis_times["CHARTTIME"] <= window_end)]
		return constants.HAS_SEPSIS if not sepsis_events.empty else constants.NO_SEPSIS
	features["label"] = features.apply(label_row, axis=1)
	print(f"Labels generated for {len(features)} rows.")
	return features

def merge_demographics(features):
	patients = pd.read_csv(os.path.join(constants.RAW_DIR, "PATIENTS.csv.gz"), 
						   usecols=["SUBJECT_ID", "GENDER", "DOB"])
	admissions = pd.read_csv(os.path.join(constants.RAW_DIR, "ADMISSIONS.csv.gz"), 
							 usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"])
	patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")
	admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
	demo = pd.merge(admissions, patients, on="SUBJECT_ID")
	demo["age"] = demo.apply(calc_age, axis=1)
	merged = pd.merge(features, demo[["SUBJECT_ID", "HADM_ID", "GENDER", "age"]], on=["SUBJECT_ID", "HADM_ID"], how="left")
	return merged

def handle_missing(features):
	numeric_cols = features.select_dtypes(include=[np.number]).columns
	features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
	return features

def preprocess():
	os.makedirs(constants.INTERMEDIATE_DIR, exist_ok=True)
	icu_admissions = pd.read_csv(os.path.join(constants.RAW_DIR, "ICUSTAYS.csv.gz"), 
								 usecols=["SUBJECT_ID", "HADM_ID", "INTIME"])
	icu_admissions["INTIME"] = pd.to_datetime(icu_admissions["INTIME"], errors="coerce")
	
	load_vitals()
	load_labs()
	features = create_features(icu_admissions)
	sepsis_times = load_sepsis_indicators(icu_admissions)
	features = generate_labels(features, icu_admissions, sepsis_times)
	features = merge_demographics(features)
	features = handle_missing(features)
	features.to_pickle(os.path.join(constants.INTERMEDIATE_DIR, "final_dataset.pkl"))
	print(f"Preprocessing complete: final dataset with {len(features)} rows saved.")

if __name__ == "__main__":
	preprocess()
