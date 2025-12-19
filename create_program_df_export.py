import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import create_student_df
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


def process_hs_directory(year):
    df_schools = pd.read_csv(f'S:/CR4239/Projects/data/hs_directories/fall-{year}.csv')
    causal_va_2023 = pd.read_csv('S:/CR4239/Projects/data/impact_performance/impact_performance_2023.csv')
    
    # standardize headings
    # format: (k,v) = (standardized heading (str), potential alternatives to replace (list))
    replacements = {'zipcode' : ['postcode', 'zip']}
    df_schools.rename(columns={old:new for new,oldlist in replacements.items() for old in oldlist}, inplace=True) 
    
    # number of APs
    df_schools['number_APs'] = (df_schools['advancedplacement_courses'].str.split(',')).str.len()
    df_schools['number_APs'] = df_schools['number_APs'].fillna(0)
    
    # causal value add metrics
    causal_HS_only = causal_va_2023[causal_va_2023['Report Type'] != 'EMS']
    df_schools = df_schools.merge(causal_HS_only[['DBN','Impact','Performance']], how='left', left_on='dbn', right_on='DBN')
    df_schools = df_schools.drop(columns='DBN')
    
    return df_schools

''' Gets program specific information from high school directory and adds to df_programs) '''
def process_hs_directory_for_programs(year, df_programs):
    df_schools = pd.read_csv(f'S:/CR4239/Projects/data/hs_directories/fall-{year}.csv')

    def extract_grade_weight(text):
        if 'Average Course Grades' in str(text):
            match = re.search(r'(\d+)%', text)
            if match:
                return int(match.group(1))/100
        return 0

    def extract_dia_fraction(text):
        if text:
            match = re.search(r'(\d+)%', str(text))
            if match:
                return int(match.group(1))/100
        return 0

    program_dict = defaultdict(lambda: {
                    'requirements':None,
                    'method':None,
                    'eligibility':None,
                    'audition_info':None,
                    'priorities':None,
                    'grade_weight':None,
                    'dia_details':None,
                    'schooldbn':None,
                    'dia_fraction':None
                })

    for index, row in df_schools.iterrows():
        for i in range(1, 13):
            if pd.notnull(row[f'code{i}']):
                programcode = str(row[f'code{i}'])

                requirements = [row[f'requirement_{j}_{i}'] for j in range(1, 5)]
                method = row[f'method{i}']
                eligibility = row[f'eligibility{i}']
                audition_info = row[f'auditioninformation{i}']
                priorities = [row[f'admissionspriority{j}{i}'] for j in range(1, 4)]
                grade_weight = extract_grade_weight(requirements[0])
                dia_details = row['diadetails']
                dia_fraction = extract_dia_fraction(dia_details)

                program_dict[programcode] = {
                    'requirements':requirements,
                    'method':method,
                    'eligibility':eligibility,
                    'audition_info':audition_info,
                    'priorities':priorities,
                    'grade_weight':grade_weight,
                    'dia_details':dia_details,
                    'dia_fraction':dia_fraction
                }
                
    for col_name in ['requirements', 'method', 'eligibility', 'audition_info', 'priorities', 'grade_weight', 'dia_details', 'dia_fraction']:
        df_programs[col_name] = df_programs['programcode'].apply(lambda x: program_dict[x][col_name])
        
    return df_programs

def create_program_to_school_dict(year):
    df_schools = process_hs_directory(year)

    progcode_cols = df_schools.filter(regex = 'code[0-9]+', axis=1).columns.to_list()
    progcode_to_schooldbn = {}
    for row in df_schools.filter(['dbn']+progcode_cols, axis=1).itertuples(index=False, name=None):
        dbn = row[0]
        for x in row[1:]:
            if isinstance(x, str) and x != 'NO CODE':
                progcode_to_schooldbn[x] = dbn  
                
    return progcode_to_schooldbn

def create_program_df(year, include_ms_tests=True):
    last_two = int(year) % 100
    cohort_directory = f'R:/CR4239/Cohort 20{last_two}-{last_two+1}/'
    df_students = pd.read_csv(cohort_directory+f'{year} HSAPS_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
    df_regs1 = pd.read_csv(cohort_directory+f'20{last_two}-{last_two+1}_Regents_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
    df_regs2 = pd.read_csv(cohort_directory+f'20{last_two+1}-{last_two+2}_Regents_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
    df_regs = pd.concat([df_regs1, df_regs2])
    if include_ms_tests:
        df_ms_tests = pd.read_csv(cohort_directory+f'20{last_two-3}-{last_two-2}_Student_test-Biog_All_G38_NYC_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
       
    df_schools = process_hs_directory(year)
    print("Number of unique schools in raw data:", df_schools['dbn'].nunique())
    progcode_to_schooldbn = create_program_to_school_dict(year)
    
    df_programs = pd.DataFrame.from_dict(progcode_to_schooldbn, orient='index', columns=['dbn']).reset_index(names='programcode')
    print('Number of unique programs in raw data:', df_programs['programcode'].nunique())
    
    # directly from schools df: zipcode, school graduation rate, number of APs, causal value add numbers
    df_programs = df_programs.merge(df_schools[['dbn','zipcode','graduation_rate', 'number_APs','Impact','Performance']], how='left', on='dbn')
    df_programs.rename(columns={'graduation_rate':'school_grad_rate', 'Impact':'impact', 'Performance':'performance'}, inplace=True)
    
    # add school quality metrics
    df_schools_quality = pd.read_csv('../data/202223-hs-sqr-with-mean.csv')
    df_schools_quality.rename(columns={'DBN':'dbn', 'Quality Review - How safe and inclusive is the school while supporting social-emotional growth':'safety'}, inplace=True)
    df_programs = df_programs.merge(df_schools_quality[['dbn', 'mean_SQR', 'safety']], how='left', on='dbn')
    
    df_programs['impact_standardized'] = (df_programs['impact'] - df_programs['impact'].mean())/df_programs['impact'].std()
    df_programs['performance_standardized'] = (df_programs['performance'] - df_programs['performance'].mean())/df_programs['performance'].std()
    df_programs['mean_SQR_standardized'] = (df_programs['mean_SQR'] - df_programs['mean_SQR'].mean())/df_programs['mean_SQR'].std()
    df_programs['aggregated_quality'] = df_programs['mean_SQR_standardized']/2 + (df_programs['impact_standardized'] + df_programs['performance_standardized'])/4
    df_programs['safe'] = df_programs['safety'] == 'Well Developed'
    
    
    # exam scores of matched students
    # for each exam, identify program where the student ended up attending
    df_regs = df_regs.merge(df_students[['student_id_scram', 'finalprogramcode']])
    df_regs.rename(columns={'finalprogramcode':'programcode'}, inplace=True)
    if include_ms_tests:
        df_ms_tests = df_ms_tests.merge(df_students[['student_id_scram', 'finalprogramcode']])
        df_ms_tests.rename(columns={'finalprogramcode':'programcode'}, inplace=True)
    # drop non-numeric regents scores
    df_regs_num = df_regs[pd.to_numeric(df_regs['mark'], errors='coerce').notnull()]
    df_regs_num['mark'] = df_regs_num['mark'].astype(float)
    # compute each student's average regents score
    df_regs_num = pd.merge(df_regs_num, 
                             df_regs_num[['mark', 'student_id_scram']].groupby('student_id_scram').mean(),
                    left_on='student_id_scram',right_index=True,how='left')
    df_regs_num.rename(columns={'mark_x':'mark','mark_y':'student_avg_mark'}, inplace=True)
    df_stud_reg_avgs = df_regs_num[['student_id_scram','programcode','student_avg_mark']].drop_duplicates(subset=['student_id_scram'])
    df_stud_reg_avgs.reset_index(drop=True, inplace=True)
    # merge into df_programs
    if include_ms_tests:
        df_programs = pd.merge(df_programs, 
                               df_ms_tests.groupby('programcode')[['ela_scale_score', 'math_scale_score']].mean(), 
                               how='left', on='programcode')
        df_programs.rename(columns={'ela_scale_score':'avg_ela_score', 'math_scale_score':'avg_math_score'}, inplace=True)
    df_programs = pd.merge(df_programs, 
                           df_regs_num.groupby('programcode')['mark'].mean(), 
                           how='left', on='programcode')
    df_programs = pd.merge(df_programs, 
                           df_stud_reg_avgs.groupby('programcode')['student_avg_mark'].mean(), 
                           how='left', on='programcode')
    df_programs.rename(columns={'mark':'avg_regents', 'student_avg_mark':'avg_student_regavg'}, inplace=True)
    
    # diversity (percentages ELL, FRL, home language)
    df_programs['final_student_count'] = df_programs['programcode'].map(df_students.groupby('finalprogramcode')['student_id_scram'].count())
    df_programs['final_student_count'] = df_programs['final_student_count'].fillna(0)
    df_programs['ell_frac'] = df_programs['programcode'].map(df_students.groupby('finalprogramcode')['ell'].mean())
    df_programs['frl_frac'] = df_programs['programcode'].map(df_students.groupby('finalprogramcode')['poverty'].mean())
    
    eth_pcts = df_students.groupby('finalprogramcode')['ethnicity'].value_counts(normalize=True).unstack()
    gen_pcts = df_students.groupby('finalprogramcode')['gender'].value_counts(normalize=True).unstack()
    lan_pcts = df_students.groupby('finalprogramcode')['homelanguagename'].value_counts(normalize=True).unstack()
    eth_pcts = eth_pcts.fillna(0).reset_index().rename(columns={'finalprogramcode':'programcode'})
    gen_pcts = gen_pcts.fillna(0).reset_index().rename(columns={'finalprogramcode':'programcode'})
    lan_pcts = lan_pcts.fillna(0).reset_index().rename(columns={'finalprogramcode':'programcode'})
    lan_pcts_concise = lan_pcts[['programcode', 'English', 'Spanish', 'Mandarin', 'Chinese/Any', 'Bengali', 'Russian', 'Arabic', 'Cantonese']]
    lan_pcts_concise['lang_frac_other'] = 1.0 - (lan_pcts['English'] + lan_pcts['Spanish'] + lan_pcts['Mandarin'] + lan_pcts['Chinese/Any'] + lan_pcts['Bengali'] + lan_pcts['Russian'] + lan_pcts['Arabic'] + lan_pcts['Cantonese'])
    lan_pcts_concise.rename(columns={'English':'lang_frac_english','Spanish':'lang_frac_spanish','Mandarin':'lang_frac_mandarin',
                             'Chinese/Any':'lang_frac_chinese_any','Bengali':'lang_frac_bengali','Russian':'lang_frac_russian',
                             'Arabic':'lang_frac_arabic','Cantonese':'lang_frac_cantonese'}, inplace=True)
    
    df_programs = pd.merge(df_programs, eth_pcts, how='left', on='programcode')
    df_programs.rename(columns={'asian':'eth_frac_asian','black':'eth_frac_black','hispanic':'eth_frac_hispanic',
                                'multiple_not_rep':'eth_frac_multiple_not_rep','white':'eth_frac_white'}, inplace=True)
    
    df_programs = pd.merge(df_programs, gen_pcts, how='left', on='programcode')
    df_programs.rename(columns={'F':'sex_frac_f','M':'sex_frac_m'}, inplace=True)
    df_programs = pd.merge(df_programs, lan_pcts_concise, how='left', on='programcode')
    
    df_programs = process_hs_directory_for_programs(year, df_programs)
    
    df_programs = compute_grad_and_college_rate(df_programs)
    
    print('Final number of unique programs:', df_programs['programcode'].nunique())
    
    return df_programs



def compute_grad_and_college_rate(df_programs):
    # cohorts where postsecondary info is now available
    years_to_include = [2017, 2018]
    dfs_to_concat = []
    
    for year in years_to_include:
        last_two = int(year) % 100
        cohort_directory = f'R:/CR4239/Cohort 20{last_two}-{last_two+1}/'
        df_cohort = create_student_df.process_hsaps(year)
        df_college = pd.read_csv(cohort_directory + f'Fall{year+4}_Postsecondary Enrollment_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
        
        # for each record in the postsecondary df, look up the program that each student attended
        df_college = pd.merge(df_college, df_cohort[['student_id_scram', 'finalprogramcode']], how='left', on='student_id_scram')
        df_college.rename(columns={'finalprogramcode':'programcode'}, inplace=True)
        
        dfs_to_concat += [df_college]
    
    df_college_all = pd.concat(dfs_to_concat, ignore_index=True)
    rates = df_college_all.groupby('programcode')[['diploma_local_or_higher','enrolled']].mean()
    
    df_programs = pd.merge(df_programs, rates, how='left', on='programcode')
    df_programs.rename(columns={'diploma_local_or_higher':'program_grad_rate', 'enrolled':'program_college_rate'}, inplace=True)
    
    return df_programs


def add_offer_rates(df_programs, df_students):
    offer_reject_dict = defaultdict(lambda: {'offers':[], 'rejects':[]})
    for index, row in df_students.iterrows():
        if pd.notnull(row['matched_choice_num']):
            matched_choice_num = int(row['matched_choice_num'])
        else:
            matched_choice_num = 13

        for i in range(1, matched_choice_num):
            rejected_program = row[f'programcode{i}']
            offer_reject_dict[rejected_program]['rejects'].append(index)
        if matched_choice_num < 13:
            offered_program = row[f'programcode{matched_choice_num}']
            offer_reject_dict[offered_program]['offers'].append(index)

    # Create a dictionary that gives the offer rate for each program
    offer_rate_dict = defaultdict(lambda: 1)
    for programcode in offer_reject_dict.keys():
        num_offers, num_rejects = len(offer_reject_dict[programcode]['offers']), len(offer_reject_dict[programcode]['rejects'])
        if num_offers + num_rejects > 0:
            offer_rate_dict[programcode] = num_offers/(num_offers + num_rejects)
        else:
            offer_rate_dict[programcode] = 1

    # Add a column in the programs df with offer rate        
    df_programs['offer_rate'] = df_programs['programcode'].apply(lambda x: offer_rate_dict[x])
    return df_programs