import pandas as pd
import torch
import src.constants as constants

def calc_age(row):
	"""Calculate age from ADMITTIME and DOB, capped at AGE_CAP."""
	intime, dob = row["ADMITTIME"], row["DOB"]
	if pd.isnull(intime) or pd.isnull(dob):
		return constants.AGE_CAP
	dt_intime, dt_dob = intime.to_pydatetime(), dob.to_pydatetime()
	if dt_dob.year > dt_intime.year:
		dt_dob = dt_dob.replace(year=dt_dob.year - constants.YEAR_SHIFT)
	age = (dt_intime - dt_dob).days / constants.DAYS_PER_YEAR
	return min(age, constants.AGE_CAP)

def save_model(model, path):
	"""Save model state to a file."""
	torch.save(model.state_dict(), path)
	print(f"Model saved to {path}")
