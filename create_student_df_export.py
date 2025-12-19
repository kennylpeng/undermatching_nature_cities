import pandas as pd
import numpy as np
import itertools
import re

'''Columns
student_id_scram: str
ethnicity: str
ell: int (1 = Yes, 0 = No)
poverty: int (1 = Yes FRL, 0 = No FRL)
gender: str
currentschooldbn: obj
resborough:
resdistrict:
ofchoices: int
zipcode: int
private_ms: boolean
charter_ms: boolean
homeschool_ms: boolean
homelang: str
ms_avg_grades: float
ms_avg_ela: float
ms_avg_math: float
ms_size: int
ms_eth_frac_asian: float
ms_eth_frac_black: float
ms_eth_frac_hispanic: float
ms_eth_frac_multiple_not_rep: float
ms_eth_frac_white: float
ms_ell_frac: float
ms_frl_frac: float
ms_sex_frac_female: float
ms_sex_frac_male: float
tract_median_age: float
tract_eth_frac_white: float
tract_eth_frac_black: float
tract_eth_frac_hispanic: float
tract_edu_frac_no_HS: float
tract_edu_frac_some_college: float
tract_edu_frac_bach_deg: float
tract_frac_renter: float
tract_frac_family: float
tract_log_med_house_val: float
tract_log_income_per_cap: float
tract_log_density: float
ela_scale_score: float
math_scale_score: float
avg_ms_grades: float
avg_regents_score: float
tiebreaker: str
tiebreaker_num: int
enrolled: boolean
cohort_year: int
'''

def process_hsaps(year):
    last_two = int(year) % 100
    cohort_directory = f'R:/CR4239/Cohort 20{last_two}-{last_two+1}/'
    df_students = pd.read_csv(cohort_directory+f'{year} HSAPS_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
        
    # standardize headings
    # format: (k,v) = (standardized heading (str), potential alternatives to replace (list))
    replacements = {'avg_marking' : ['avg_grades'], 'finalprogramcode' : ['final_disposition']}
    df_students.rename(columns={old:new for new,oldlist in replacements.items() for old in oldlist}, inplace=True) 
    
    # standardize all non-matches as nan's
    df_students.replace('-', np.nan)
    
    return df_students

def create_student_df(year, include_ms_tests=True, include_enrollment=True):   
    last_two = int(year) % 100
    cohort_directory = f'R:/CR4239/Cohort 20{last_two}-{last_two+1}/'
    df_students = process_hsaps(year)
    
    print('Unique students in raw data:', df_students['student_id_scram'].nunique())
    
    df_regs1 = pd.read_csv(cohort_directory+f'20{last_two}-{last_two+1}_Regents_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
    df_regs2 = pd.read_csv(cohort_directory+f'20{last_two+1}-{last_two+2}_Regents_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
    df_regs = pd.concat([df_regs1, df_regs2])
    
    print('Number of Regents exam scores loaded:', df_regs.shape[0])
    
    if include_ms_tests: 
        df_ms_tests = pd.read_csv(cohort_directory+f'20{last_two-3}-{last_two-2}_Student_test-Biog_All_G38_NYC_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
        print('Number of middle school exam scores loaded:', df_ms_tests.shape[0])
    
    # census tract demographics -- using latest (2024 Q2) mapping, and demographics corresponding to application year
    if year < 2019: demoyear = 2019
    elif year > 2022: demoyear = 2022
    else: demoyear = year
    print('Census tract demographics year:', demoyear)
    file_tract_demo = f'S:/CR4239/Projects/data/zip_census_data/tract_demographics/{demoyear}.csv'
    df_crosswalk = pd.read_csv('S:/CR4239/Projects/data/zip_census_data/HUD-ZIP_to_tract/2024_quarter-2.csv')
    df_tractdemo = pd.read_csv(file_tract_demo)
    df_crosswalk['geoid_ratio_pair'] = list(zip(df_crosswalk.geoid, df_crosswalk.res_ratio))
    
    ##### HELPER METHODS #####
    def top_lang_or_other(langname, top_k=8):  
        toplangs = df_students.homelanguagename.value_counts()[0:top_k].index
        if langname in toplangs: return langname
        return 'Other'

    def is_private(school_dbn):
        if pd.notnull(school_dbn):
            if school_dbn[-1]=='P':
                return True
            else:
                return False
        else:
            return False

    def is_charter(school_dbn):
        if pd.notnull(school_dbn):
            if school_dbn[0:2]=='84':
                return True
            else:
                return False
        else:
            return False
    
    def is_homeschool(school_dbn):
        if pd.notnull(school_dbn):
            if school_dbn[-3:]=='444':
                return True
            else:
                return False
        else:
            return False
        
    def get_tiebreaker_num(tiebreaker):
        if type(tiebreaker)==str:
            head = tiebreaker[:5]
            return int(format(int(head, 16), 'd'))
        else:
            return np.nan
    ######################
    
    df_stud_export = df_students[['student_id_scram', 'ethnicity', 'ell', 'poverty', 'gender',
                                  'type', 'edoptcategory',
                                  'currentschooldbn', 'ofchoices', 'resborough']]
    df_stud_export['zip_code'] = df_students['zip_code'].fillna(0).astype(np.int64)
    df_stud_export.rename(columns={'zip_code':'zipcode'}, inplace=True)
    df_stud_export['private_ms'] = df_students['currentschooldbn'].apply(lambda dbn: is_private(dbn))
    df_stud_export['charter_ms'] = df_students['currentschooldbn'].apply(lambda dbn: is_charter(dbn))
    df_stud_export['homeschool_ms'] = df_students['currentschooldbn'].apply(lambda dbn: is_homeschool(dbn))
    df_stud_export['homelang'] = df_students['homelanguagename'].apply(lambda l: top_lang_or_other(l))
    
    # student's (individual) performance in middle school
    if include_ms_tests:
        df_students = pd.merge(df_students, df_ms_tests[['student_id_scram', 'ela_scale_score', 'math_scale_score']], how='left', on='student_id_scram')
        df_stud_export['ela_scale_score'] = df_students['ela_scale_score']
        df_stud_export['math_scale_score'] = df_students['math_scale_score']
    df_stud_export['avg_ms_grades'] = df_students['avg_marking']
    df_stud_export['avg_grades_for_tier'] = df_students['avg_marking']
    

    # student's (individual) high school outcomes
    # for each exam, identify program where the student ended up attending
    df_regs = pd.merge(df_regs, df_students[['student_id_scram', 'finalprogramcode']], how='left', on='student_id_scram')
    df_regs.rename(columns={'finalprogramcode':'programcode'}, inplace=True)
    # drop non-numeric regents scores
    df_regs_num = df_regs[pd.to_numeric(df_regs['mark'], errors='coerce').notnull()]
    df_regs_num['mark'] = df_regs_num['mark'].astype(float)
    # compute each student's average regents score
    df_regs_num = pd.merge(df_regs_num, df_regs_num[['mark', 'student_id_scram']].groupby('student_id_scram').mean(), on='student_id_scram',how='left')
    # mark_x --> exam_mark is this particular exam score; mark_y --> student_avg_mark is this particular student's average exam score
    df_regs_num.rename(columns={'mark_x':'exam_mark','mark_y':'student_avg_mark'}, inplace=True)    
    # students are listed multiple times (once for each exam), but each student only is associated with one progcode and one avg_mark, so we can just drop duplicates
    # e.g. 2021 cohort: 306829 exams --> 59970 unique students
    df_stud_reg_avgs = df_regs_num[['student_id_scram','programcode','student_avg_mark']].drop_duplicates(subset=['student_id_scram'])
    print('Number of students with numerical Regents averages:', df_stud_reg_avgs.shape[0])
    df_stud_reg_avgs.reset_index(drop=True, inplace=True)
    
    df_students = df_students.merge(df_stud_reg_avgs[['student_id_scram', 'student_avg_mark']], how='left', on='student_id_scram')
    df_students.rename(columns={'student_avg_mark':'regavg'}, inplace=True)
    
    df_stud_export['avg_regents_score'] = df_students['regavg']
    
    # average performance of student's middle school
    ms_avg_marking = df_students.groupby('currentschooldbn')[['avg_marking']].mean()
    df_stud_export['ms_avg_grades'] = df_stud_export['currentschooldbn'].map(ms_avg_marking['avg_marking'])
    if include_ms_tests:
        ms_avg_tests = df_students.groupby('currentschooldbn')[['ela_scale_score', 'math_scale_score']].mean()
        df_stud_export['ms_avg_ela'] = df_stud_export['currentschooldbn'].map(ms_avg_tests['ela_scale_score'])
        df_stud_export['ms_avg_math'] = df_stud_export['currentschooldbn'].map(ms_avg_tests['math_scale_score'])
    
    # middle school demographics
    ms_counts = df_students.groupby('currentschooldbn').size()
    ms_eth_pcts = df_students.groupby('currentschooldbn')['ethnicity'].value_counts(normalize=True).unstack()
    ms_ell_pcts = df_students.groupby('currentschooldbn')['ell'].value_counts(normalize=True).unstack()
    ms_pov_pcts = df_students.groupby('currentschooldbn')['poverty'].value_counts(normalize=True).unstack()
    ms_gen_pcts = df_students.groupby('currentschooldbn')['gender'].value_counts(normalize=True).unstack()

    df_stud_export['ms_size'] = df_stud_export['currentschooldbn'].map(ms_counts)
    df_stud_export['ms_eth_frac_asian'] = df_stud_export['currentschooldbn'].map(ms_eth_pcts['asian'])
    df_stud_export['ms_eth_frac_black'] = df_stud_export['currentschooldbn'].map(ms_eth_pcts['black'])
    df_stud_export['ms_eth_frac_hispanic'] = df_stud_export['currentschooldbn'].map(ms_eth_pcts['hispanic'])
    df_stud_export['ms_eth_frac_multiple_not_rep'] = df_stud_export['currentschooldbn'].map(ms_eth_pcts['multiple_not_rep'])
    df_stud_export['ms_eth_frac_white'] = df_stud_export['currentschooldbn'].map(ms_eth_pcts['white'])
    df_stud_export['ms_ell_frac'] = df_stud_export['currentschooldbn'].map(ms_ell_pcts[1])
    df_stud_export['ms_frl_frac'] = df_stud_export['currentschooldbn'].map(ms_pov_pcts[1])
    df_stud_export['ms_sex_frac_female'] = df_stud_export['currentschooldbn'].map(ms_gen_pcts['F'])
    df_stud_export['ms_sex_frac_male'] = df_stud_export['currentschooldbn'].map(ms_gen_pcts['M'])
    
    # census tract demographics 
    # define a function mapping lists to weighted averages of each tract
    def compute_avg_from_list(tract_ratio_list, feature, data=df_tractdemo):
        result = 0
        if isinstance(tract_ratio_list, list):  # ignore NaNs or 0s or whatever else
            for (tract, ratio) in tract_ratio_list:
                if tract in data.index:    # handle weird missing tracts
                    result += data.loc[tract][feature] * ratio
        return result
    
    df_tractdemo['frac_white'] = df_tractdemo['total_white'] / df_tractdemo['total_pop']
    df_tractdemo['frac_black'] = df_tractdemo['total_black'] / df_tractdemo['total_pop']
    df_tractdemo['frac_hispanic'] = df_tractdemo['total_hispanic'] / df_tractdemo['total_pop']
    df_tractdemo['frac_poverty'] = df_tractdemo['total_poverty'] / df_tractdemo['total_pop']
    df_tractdemo['frac_renter'] = df_tractdemo['total_renter'] / df_tractdemo['total_pop']
    df_tractdemo['frac_no_HS'] = df_tractdemo['total_no_HS'] / df_tractdemo['total_pop']
    df_tractdemo['frac_some_college'] = df_tractdemo['total_some_col_assoc'] / df_tractdemo['total_pop']
    df_tractdemo['frac_bach_deg'] = df_tractdemo['total_bach'] / df_tractdemo['total_pop']
    df_tractdemo['frac_family'] = (df_tractdemo['total_married_fam'] + df_tractdemo['total_other_fam']) / df_tractdemo['total_pop']
    df_tractdemo['log_med_house_val'] = np.log(df_tractdemo['med_house_val'])
    df_tractdemo['log_income_per_cap'] = np.log(df_tractdemo['per_cap_income'])
    df_tractdemo['log_density'] = np.log(df_tractdemo['total_pop'] / df_tractdemo['area_m2'])
    df_tractdemo.update(df_tractdemo.select_dtypes(include=[np.number]).fillna(0))
    df_tractdemo.set_index('GEOID', inplace=True)
    
    zip_to_tracts = df_crosswalk.groupby('zip')['geoid_ratio_pair'].apply(list)
    df_stud_export['tracts_and_ratios'] = df_stud_export['zipcode'].map(zip_to_tracts)

    df_stud_export['tract_median_age'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'med_age'))
    df_stud_export['tract_eth_frac_white'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_white'))
    df_stud_export['tract_eth_frac_black'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_black'))
    df_stud_export['tract_eth_frac_hispanic'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_hispanic'))
    df_stud_export['tract_edu_frac_no_HS'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_no_HS'))
    df_stud_export['tract_edu_frac_some_college'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_some_college'))
    df_stud_export['tract_edu_frac_bach_deg'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_bach_deg'))
    df_stud_export['tract_frac_renter'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_renter'))
    df_stud_export['tract_frac_family'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'frac_family'))
    df_stud_export['tract_log_med_house_val'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'log_med_house_val'))
    df_stud_export['tract_log_income_per_cap'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'log_income_per_cap'))
    df_stud_export['tract_log_density'] = df_stud_export['tracts_and_ratios'].apply(lambda trl: compute_avg_from_list(trl, 'log_density'))

    df_stud_export.drop(columns=['tracts_and_ratios'], inplace=True)
    
    # tiebreaker (lottery number)
    df_stud_export['tiebreaker'] = df_students['tiebreaker']
    df_stud_export['tiebreaker_num'] = df_stud_export['tiebreaker'].apply(lambda tb: get_tiebreaker_num(tb))
    
    
    # application portfolio
    df_stud_export = pd.merge(df_stud_export, df_students[['student_id_scram']+[f'programcode{i}' for i in range(1,13)]],
                              on='student_id_scram', how='left')
    
    # match outcomes (no counterfactuals yet)
    df_stud_export = pd.merge(df_stud_export, df_students[['student_id_scram', 'matched', 'matched_program', 'matched_dbn', 'matched_choice_num',
                                                             'mp', 'mp_programcode', 'mp_schooldbn',
                                                             'finalprogramcode', 'finalschooldbn']], 
                              on='student_id_scram', how='left')
    
    df_stud_export = df_stud_export.merge(df_students[['student_id_scram', 'shs_offer', 'lga_offer', 'private_no_enroll', 
                                                          'shsat_offer_programcode', 'lga_offer_programcode1']], 
                                             on='student_id_scram', how='left')
    
    # enrollment (i.e. received a grade the following school year)
    if include_enrollment:
        # get list of students who received a grade in the following school year
        df_enroll = pd.read_csv(cohort_directory+f'20{last_two}-{last_two+1}_HsCourseAndGrades_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
        print("Number of unique students in subsequent year enrollment data:", df_enroll['student_id_scram'].nunique())
        df_enroll_unique = df_enroll.drop_duplicates(subset=['student_id_scram'])  # just need one grade per student
        enrollment_dict = dict(zip(df_enroll_unique['student_id_scram'], df_enroll_unique['dbn']))
        df_stud_export['enrolled'] = df_stud_export['student_id_scram'].apply(lambda x: x in enrollment_dict.keys())
    
        # verify finalprogramcode against enrollment from next year's grades (mostly relevant for SHS/LGA)
        def find_enrolled_progcode(row):
            if pd.notna(row['finalprogramcode']) and re.fullmatch(r'[a-zA-Z0-9]{4}', row['finalprogramcode']):
                return row['finalprogramcode']
                
            # otherwise, finalprogramcode is NaN or something like "-" -- check for SHS/LGA
            if row['shs_offer'] == 1:
                return row['shsat_offer_programcode']
            elif row['lga_offer'] == 1:
                return row['lga_offer_programcode1']  # some (but very few) students receive multiple LGA offers -- will just assume they take offer 1
                
            # otherwise should be NaN
            return row['finalprogramcode']
            
        df_stud_export['enrolled_programcode'] = df_stud_export.apply(lambda row: find_enrolled_progcode(row), axis=1)
        df_stud_export.drop(columns=['shsat_offer_programcode', 'lga_offer_programcode1'], inplace=True)  # now redundant
        print('Number of students where finalprogramcode != enrolled_programcode:', 
              df_stud_export[df_stud_export['finalprogramcode'] != df_stud_export['enrolled_programcode']]['enrolled_programcode'].count())
        
    # add cohort year
    df_stud_export['cohort_year'] = year

    df_stud_export = df_stud_export.drop_duplicates(subset = 'student_id_scram')
    
    # filter for students who apply to at least one school, and GE only
    df_stud_export = df_stud_export[df_stud_export['ofchoices'] >= 1]
    df_stud_export = df_stud_export[df_stud_export['type'] == 'GE']
    
    # standardize all non-matches as nan's
    df_stud_export.loc[~df_stud_export['matched_program'].str.contains(r'[A-Za-z0-9]{4}$', na=False), 'matched_program'] = np.nan
    df_stud_export.loc[~df_stud_export['finalprogramcode'].str.contains(r'[A-Za-z0-9]{4}$', na=False), 'finalprogramcode'] = np.nan
    
    print('Final unique student count:', df_stud_export['student_id_scram'].nunique())
    
    return df_stud_export