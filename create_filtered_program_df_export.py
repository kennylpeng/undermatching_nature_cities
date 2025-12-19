import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import re

'''Columns
programcode: str
dbn: str
zipcode: int
school_grad_rate: float
number_APs: float
impact: float 
performance: float
avg_ela_score: float
avg_math_score: float
avg_regents: float
avg_student_regavg: float
final_student_count: float
ell_frac: float
frl_frac: float
eth_frac_asian: float
eth_frac_black: float
eth_frac_hispanic: float
eth_frac_multiple_not_rep: float
eth_frac_white: float
sex_frac_f: float
sex_frac_m: float
lang_frac_english: float
lang_frac_spanish: float
lang_frac_mandarin: float
lang_frac_chinese_any: float
lang_frac_bengali: float
lang_frac_russian: float
lang_frac_arabic: float
lang_frac_cantonese: float
lang_frac_other: float
program_grad_rate: float
program_college_rate: float


'requirements': list<str>,
'method': str,
'eligibility': list<str>,
'audition_info': list<str>,
'priorities': list<str>,
'grade_weight': float,
'dia_details': str,
'dia_fraction': float
'''


''' Filters df_programs for only programs with applications AND counterfactuals.
Also adds standardized impact and performance metrics (# of std devs away from mean).'''
def create_filtered_program_df(df_programs, df_students):
    print('Number of unique programs in raw data:', df_programs.programcode.nunique())
    print('Number of unique schools in raw data:', df_programs.dbn.nunique())
    
    # filter for programs where a student applied
    progs_with_apps = set(df_students[['programcode1',
                         'programcode2',
                         'programcode3',
                         'programcode4',
                         'programcode5',
                         'programcode6',
                         'programcode7',
                         'programcode8',
                         'programcode9',
                         'programcode10',
                         'programcode11',
                         'programcode12']].values.flatten())
    progs_with_apps.remove(np.nan)
    
    df_progs_applied = df_programs[df_programs['programcode'].isin(progs_with_apps)]
    print('Number of unique programs with apps:', df_progs_applied.programcode.nunique())
    print('Number of unique schools with apps:', df_progs_applied.dbn.nunique())
    
    # filter for programs with a counterfactual
    df_progs_filtered = df_progs_applied[df_progs_applied['counterfactual']==True]
    print('Number of unique programs with apps and cfs:', df_progs_filtered.programcode.nunique())
    print('Number of unique schools with apps and cfs:', df_progs_filtered.dbn.nunique())
    
    # compute mean and std dev on the school-level 
    df_school_quality = df_progs_filtered.drop_duplicates(subset=['dbn'])[['dbn', 'impact', 'performance']]
    mean_impact = df_school_quality['impact'].mean()
    sd_impact = df_school_quality['impact'].std()
    mean_performance = df_school_quality['performance'].mean()
    sd_performance = df_school_quality['performance'].std()
    
    # standardize on the program-level
    df_progs_filtered['impact_standardized'] = (df_progs_filtered['impact'] - mean_impact) / sd_impact
    df_progs_filtered['performance_standardized'] = (df_progs_filtered['performance'] - mean_performance) / sd_performance
    
    # deduplicate
    df_progs_filtered.drop_duplicates(subset=['programcode'], inplace=True)
    
    return df_progs_filtered


''' Filters df_programs for only programs that received an application from a student in df_students. 
Also adds standardized impact and performance metrics (# of std devs away from mean).'''
def create_programs_with_apps_df(df_programs, df_students):
    print('Number of unique programs in raw data:', df_programs.programcode.nunique())
    print('Number of unique schools in raw data:', df_programs.dbn.nunique())
    
    # filter for programs where a student applied
    progs_with_apps = set(df_students[['programcode1',
                         'programcode2',
                         'programcode3',
                         'programcode4',
                         'programcode5',
                         'programcode6',
                         'programcode7',
                         'programcode8',
                         'programcode9',
                         'programcode10',
                         'programcode11',
                         'programcode12']].values.flatten())
    progs_with_apps.remove(np.nan)
    
    df_progs_applied = df_programs[df_programs['programcode'].isin(progs_with_apps)]
    print('Number of unique programs with apps:', df_progs_applied.programcode.nunique())
    print('Number of unique schools with apps:', df_progs_applied.dbn.nunique())
    
    # compute mean and std dev on the school-level 
    df_school_quality = df_progs_applied.drop_duplicates(subset=['dbn'])[['dbn', 'impact', 'performance']]
    mean_impact = df_school_quality['impact'].mean()
    sd_impact = df_school_quality['impact'].std()
    mean_performance = df_school_quality['performance'].mean()
    sd_performance = df_school_quality['performance'].std()
    
    # standardize on the program-level
    df_progs_applied['impact_standardized'] = (df_progs_applied['impact'] - mean_impact) / sd_impact
    df_progs_applied['performance_standardized'] = (df_progs_applied['performance'] - mean_performance) / sd_performance
    
    # deduplicate
    df_progs_applied.drop_duplicates(subset=['programcode'], inplace=True)
    
    return df_progs_applied
