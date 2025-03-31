# Sepsis Prediction Model
This project aims to predict the likelihood of sepsis onset in ICU patients within a 24-hour window using time-windowed features from MIMIC-III data. It includes preprocessing, analysis, training, and evaluation.<br>

NOTE: Make sure the path to the MIMIC-III data is copied to `RAW_DIR` in constants.py<br>

To run, install packages from requirements.txt (I used Python 3.12.9) and run the following commands:<br>
`python -m src.preprocess`<br>
`python -m src.analyze`<br>
`python -m src.train`<br>
