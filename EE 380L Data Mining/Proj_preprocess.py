import os

import numpy as np
import pandas as pd

import argparse
import progressbar

parser = argparse.ArgumentParser(description='Code to preprocess data from the eICU database')
parser.add_argument('--path', help='Path to eICU database', required=True, type=str)
args = parser.parse_args()

assert len(args.path) > 0, 'Empty path'

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('StayID')
            ]

# Typical values for imputation, from Benchmarking ML algorithms paper.
impute_values = {'Eyes': 4, 'GCS Total': 15, 'Heart Rate': 86, 'Motor': 6, 'Invasive BP Diastolic': 56,
                     'Invasive BP Systolic': 118, 'O2 Saturation': 98, 'Respiratory Rate': 19,
                     'Verbal': 5, 'glucose': 128, 'admissionweight': 81, 'Temperature (C)': 36,
                     'admissionheight': 170, "MAP (mmHg)": 77, "pH": 7.4, "FiO2": 0.21}

logfile = open(os.path.join('temp', 'preprocess.log'), 'w')

# Read patients.csv
patients = pd.read_csv(os.path.join(args.path, 'patient.csv.gz'), compression='gzip')

logfile.write("patients has {} records\n".format(patients.shape[0]))

# Only choose relevant columns
patients = patients[['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx', 'admissionheight', 'admissionweight', 'dischargeweight', 'hospitaladmitoffset', 'hospitaldischargeoffset', 'unitdischargeoffset', 'uniquepid', 'hospitaldischargestatus', 'unitdischargestatus']]

# Filter patients by age
patients = patients.loc[patients['age'] != '> 89']
patients = patients.astype({'age': 'float'})
patients = patients.loc[(patients['age'] >= 18) & (patients['age'] <= 89)]

logfile.write("patients has {} records after filtering by age\n".format(patients.shape[0]))

# Filter patients by number of stays
id_counts = patients.groupby(by='uniquepid').count()
single_visit_ids = id_counts[id_counts['patientunitstayid'] == 1].index
patients = patients.loc[patients['uniquepid'].isin(single_visit_ids)]

logfile.write("patients has {} records after filtering by number of stays\n".format(patients.shape[0]))

# Filter patients by gender
gender_map = {'Female': 1, 'Male': 2}
patients = patients.loc[patients['gender'].isin(gender_map)] # Removes records having unknown gender
patients['gender'] = patients['gender'].map(gender_map)

logfile.write("patients has {} records after filtering by gender\n".format(patients.shape[0]))

# Filter patients by discharge status
discharge_map = {'Alive': 0, 'Expired': 1}
patients = patients.loc[patients['hospitaldischargestatus'].isin(discharge_map)]
patients = patients.loc[patients['unitdischargestatus'].isin(discharge_map)]

patients['hospitaldischargestatus'] = patients['hospitaldischargestatus'].map(discharge_map)
patients['unitdischargestatus'] = patients['unitdischargestatus'].map(discharge_map)

logfile.write("patients has {} records after filtering by discharge status\n".format(patients.shape[0]))

# Convert ethnicity to numbers
ethnicity_map = {'Asian': 1, 'African American': 2, 'Caucasian': 3, 'Hispanic': 4, 'Native American': 5, 'NaN': 0, '': 0}
patients.update({'ethnicity': patients['ethnicity'].fillna('').apply(lambda s: ethnicity_map[s] if s in ethnicity_map else ethnicity_map[''])})

logfile.write("patients has {} records after filtering by ethnicity\n".format(patients.shape[0]))

# Convert diagnoses to numbers
patients['apacheadmissiondx'].fillna('nodx', inplace=True)
dx_vals, dx_keys = pd.factorize(patients['apacheadmissiondx'].unique())
apacheadmissiondx_map = dict(zip(dx_keys, dx_vals))
patients['apacheadmissiondx'] = patients['apacheadmissiondx'].map(apacheadmissiondx_map)

logfile.write("patients has {} records after filtering by diagnosis\n".format(patients.shape[0]))

# Using the average of admission and discharge weight wherever possible
patients.loc[patients['dischargeweight'].notnull(), 'admissionweight'] = 0.5*(patients['admissionweight'] + patients['dischargeweight'])

# Clip values to range
patient_features = ['admissionheight', 'admissionweight']
patient_feature_ranges = [(100, 240), (30, 250)]
for feature, (minval, maxval) in zip(patient_features, patient_feature_ranges):
    patients[feature].clip(minval, maxval, inplace=True)

# Drop unnecessary columns
patients.drop(columns=['dischargeweight', 'hospitaladmitoffset', 'hospitaldischargeoffset'], inplace=True)

# Select stayids
stayids = patients['patientunitstayid']

patients.to_csv(os.path.join(args.path, 'patient_features.csv.gz'), compression='gzip')
del patients

'''
patients = pd.read_csv(os.path.join(args.path, 'patient_features.csv.gz'), compression='gzip')
stayids = patients['patientunitstayid']

del patients
'''

# Read nurseCharting.csv
nursingchart = pd.read_csv(os.path.join(args.path, 'nurseCharting.csv'))

logfile.write('Loaded nurseCharting\n')

# Drop unnecessary columns
nursingchart.drop(['nursingchartentryoffset','nursingchartcelltypecat'],axis=1,inplace=True)

# Only select relevant rows
nursingchart = nursingchart[nursingchart['patientunitstayid'].isin(stayids)]

# Rename columns for convenience
nursingchart.rename(index=str, columns={"nursingchartoffset": "offset",
                                  "nursingchartcelltypevalname": "itemname",
                                  "nursingchartcelltypevallabel": "itemlabel",
                                  "nursingchartvalue": "itemvalue"}, inplace=True)

logfile.write('Renamed nurseCharting columns\n')

# Select features of interest and keys
nursingchart_featurelabels = ['Heart Rate','MAP (mmHg)','Arterial Line MAP (mmHg)']
nursingchart_featurenames = ['Non-Invasive BP Systolic', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic','Invasive BP Diastolic', 'GCS Total', 'Verbal', 'Eyes', 'Motor', 'O2 Saturation', 'Respiratory Rate', 'Temperature (F)']
nursingchart = nursingchart[(nursingchart.itemlabel.isin(nursingchart_featurelabels)) | (nursingchart.itemname.isin(nursingchart_featurenames))]

logfile.write('Selected rows containing features of interest from nurseCharting\n')

# Unify different names from the itemname and labels here to just have 4 features
nursingchart.loc[nursingchart['itemname'] == 'Value', 'itemname'] = nursingchart['itemlabel'] # Replace itemlabel of 'Value' with itemnames 
nursingchart.loc[nursingchart['itemname'] == 'Non-Invasive BP Systolic', 'itemname'] = 'Invasive BP Systolic'# Unify Non invasive to invase systolic
nursingchart.loc[nursingchart['itemname'] == 'Non-Invasive BP Diastolic', 'itemname'] = 'Invasive BP Diastolic'# Unify Non invasive to invase diastolic
nursingchart.loc[nursingchart['itemlabel'] == 'Arterial Line MAP (mmHg)', 'itemname'] = 'MAP (mmHg)'# Unify Arterial MAP and MAP to single MAP

logfile.write('Unified features in nurseCharting\n')

# Drop item label after unifying names
nursingchart.drop(['itemlabel','nursingchartid'],axis=1,inplace=True)

# Converting key-value pairs to new columns
nursingchart = nursingchart.pivot_table(index=['patientunitstayid','offset'], columns='itemname', values='itemvalue',aggfunc='first').reset_index()
logfile.write('Converted key-value pairs to columns in nurseCharting\n')

nursingchart['GCS Total'] = nursingchart['GCS Total'].map({'Unable to score due to medication': np.nan})

logfile.write('Converted key-value pairs to columns in nurseCharting\n')

# Cast table to float
nursingchart = nursingchart.astype('float')

# Convert Fahrenheit to Celsius
nursingchart['Temperature (F)'] = (nursingchart['Temperature (F)'] - 32)*(5/9)
nursingchart.rename(index=str, columns={'Temperature (F)': 'Temperature (C)'}, inplace=True)

logfile.write('Converted Fahrenheit to Celsius in nurseCharting\n')

# Clip values to range
nursingchart_features = ['Invasive BP Diastolic', 'Invasive BP Systolic', 'Heart Rate', 'MAP (mmHg)', 'GCS Total', 'Verbal', 'Eyes', 'Motor', 'O2 Saturation', 'Respiratory Rate', 'Temperature (C)']
nursingchart_feature_ranges = [(0, 375), (0, 375), (0, 350), (14, 330), (2, 16), (1, 5), (0, 5), (0, 6), (0, 100), (0, 100), (26, 45)]
for feature, (minval, maxval) in zip(nursingchart_features, nursingchart_feature_ranges):
    nursingchart[feature].clip(minval, maxval, inplace=True)

# Bin offsets into hours
nursingchart['offset'] = (nursingchart['offset']/60).astype('int')
# Impute values within offset by replacing NaN with mean over each column.
nursingchart.groupby(['patientunitstayid', 'offset']).apply(lambda x: x.fillna(x.mean()))
# For each offset, only choose last value.
nursingchart.drop_duplicates(['patientunitstayid', 'offset'], keep='last', inplace=True)
# Impute missing values with "typical values"
nursingchart.fillna(value=impute_values, inplace=True)

logfile.write('Binned and imputed nurseCharting features')

nursingchart.to_csv(os.path.join(args.path, 'nursingchart_features.csv.gz'), compression='gzip')

logfile.write('Wrote nurseCharting features to CSV\n')

del nursingchart

lab = pd.read_csv(os.path.join(args.path, 'lab.csv.gz'), compression='gzip')

logfile.write('Loaded lab\n')

# Only select relevant columns
lab = lab.[['patientunitstayid', 'labresultoffset', 'labname', 'labresult']]

# Only select relevant rows
lab = lab[lab['patientunitstayid'].isin(stayids)]

# Rename columns for convenience
lab.rename(index=str, columns={"labresultoffset": "offset",
                               "labname": "itemname",
                               "labresult": "itemvalue"}, inplace=True)

logfile.write('Renamed lab columns\n')

# Select features of interest and keys
lab_featurenames = ['glucose', 'bedside glucose', 'pH', 'FiO2']
lab = lab[(lab.itemname.isin(lab_featurenames))]

logfile.write('Selected rows of interest from lab\n')

# Unify bedside glucose and glucose
lab.loc[lab['itemname'] == 'bedside glucose', 'itemname'] = 'glucose'

logfile.write('Unified features in lab\n')

# Convert key-value pairs to new columns
lab = lab.pivot_table(index=['patientunitstayid','offset'], columns='itemname', values='itemvalue',aggfunc='first').reset_index()

logfile.write('Converted key-value pairs to columns in lab\n')

# Casting columns to float
lab = lab.astype('float')
lab['FiO2'] = lab['FiO2']/100

# Clip values to range
lab_features = ['glucose', 'pH', 'FiO2']
lab_feature_ranges = [(33, 1200), (6.3, 10), (15, 110)]
for feature, (minval, maxval) in zip(lab_features, lab_feature_ranges):
    lab[feature].clip(minval, maxval, inplace=True)

# Bin offsets into hours
lab['offset'] = (lab['offset']/60).astype('int')

# Impute values within offset by replacing NaN with mean over each column.
lab.groupby(['patientunitstayid', 'offset']).apply(lambda x: x.fillna(x.mean()))

# For each offset, only choose last value.
lab.drop_duplicates(['patientunitstayid', 'offset'], keep='last', inplace=True)

# Impute missing values with "typical values"
lab.fillna(value=impute_values, inplace=True)

logfile.write('Binned and imputed features from lab\n')

lab.to_csv(os.path.join(args.path, 'lab_features.csv.gz'), compression='gzip')

logfile.write('Wrote lab features to CSV\n')

del lab

# Combining all features
patients = pd.read_csv(os.path.join(args.path, 'patient_features.csv.gz'), compression='gzip')
nursingchart = pd.read_csv(os.path.join(args.path, 'nursingchart_features.csv.gz'), compression='gzip')
lab = pd.read_csv(os.path.join(args.path, 'lab_features.csv.gz'), compression='gzip')

temp = pd.merge(nc, lab, how='outer', on=['patientunitstayid', 'offset']).sort_values(by=['patientunitstayid', 'offset'])
all_features = pd.merge(temp, patients, how='outer', on='patientunitstayid').sort_values(by=['patientunitstayid', 'offset'])

# Impute missing values with "typical values"
all_features.fillna(value=impute_values, inplace=True)

# Filter by number of records
all_features = all_features.groupby('patientunitstayid').filter(lambda x: (x.shape[0] >= 15 and x.shape[0] <= 200))

# Compute RLOS
all_features['rlos'] = all_features['unitdischargeoffset']/1440 - res['offset']/24

# Only choose records having positive offsets and RLOS
all_features = all_features[all_features['offset'] > 0]
all_features = all_features[(all_features['unitdischargeoffset'] > 0) & (res['rlos'] > 0)]

# Write features to CSV
all_features.to_csv(os.path.join(args.path, 'eicu_features.csv.gz'), compression='gzip')

logfile.write('Wrote all features to CSV\n')
logfile.close()
