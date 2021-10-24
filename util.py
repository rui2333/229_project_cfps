import pandas as pd
import glob
from pathlib import Path
import os


def load_dta(dta_path):
    """Load dataset into a pandas dataframe"""

    data=pd.read_stata(dta_path,
                       convert_categoricals=False,
                       convert_missing=True,
                       preserve_dtypes=False
                      )
    return data

def generate_dta_path(file_type):
    """Generate the relative path to the dta file"""

    return Path('.').rglob(f'*{file_type}*.dta')

def write_to_csv(df, file_name):
    """Write data to csv file"""
    df.to_csv(file_name, sep='\t', encoding='utf-8')
    
if __name__ == "__main__":
    data_frames = [load_dta(dta_file) for dta_file in generate_dta_path('person')]
    for df in data_frames:
        print(df.shape)