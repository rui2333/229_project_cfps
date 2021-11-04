import pandas as pd
import numpy as np
from pathlib import Path
import sys


y_cols = ['qn406','qn407','qn411','qn412','qn414','qn416','qn418','qn420']
personal_info_cols = ['pid', 'code', 'fid18', 'fid16', 'fid14', 'fid12', 'fid10', 'pid_a_f', 'pid_a_m']

def load_dta(dta_path):
    """Load dataset into a pandas dataframe"""

    data=pd.read_stata(dta_path,
                       convert_categoricals=False,
                       preserve_dtypes=False
                      )
    return data

def generate_dta_path(file_type):
    """Generate the relative path to the dta file"""

    return Path('.').rglob(f'*{file_type}*.dta')

def write_to_csv(df, file_name):
    """Write data to csv file"""
    df.to_csv(file_name, sep='\t', encoding='utf-8')

def drop_personal_info(data):
    """Drop personal info related columns"""
    return data.drop(personal_info_cols, axis=1)

def clean_up_na_values(data):
    """Remove all the rows that contains nan value"""
    data.replace('', np.nan, inplace=True)
    ground_truth, _ = split_groud_truth(data)
    data = data.dropna(axis=1)
    data = data.join(ground_truth)
    data = data.dropna()
    return data

def split_groud_truth(data):
    """Separate ground truth with traning features"""
    return data[y_cols], data.drop(y_cols, axis=1)
    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        data_frames = [load_dta(dta_file) for dta_file in generate_dta_path(sys.argv[1])]
        for df in data_frames:
            df = drop_personal_info(df)
            df = clean_up_na_values(df)
            ground_truth_df, training_df = split_groud_truth(df)
            print(ground_truth_df.shape)
            print(training_df.shape)