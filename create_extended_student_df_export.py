import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from collections import Counter

def create_extended_student_df(df_students, df_programs, df_counterfactuals, programs_counterfactual, drop_helpers=True, impute_clusters=True):
    df_zipzip = pd.read_csv('S:\CR4239\Projects\zipzipdf.csv')
    
    print('Full length of df_students:', df_students.shape[0])
    print('Unique students in df_students:', df_students['student_id_scram'].nunique())
    
    # drop duplicates
    df_students = df_students.drop_duplicates(subset = 'student_id_scram')
    
    # filter for students who apply to at least one school, and GE only
    df_students = df_students[df_students['ofchoices'] >= 1]
    df_students = df_students[df_students['type'] == 'GE']
    print('Unique students who apply to at least one school, GE-only:', df_students['student_id_scram'].nunique())
    
    # join in counterfactuals for portfolio/regret computations
    df_students_cf = add_counterfactuals(df_students, df_counterfactuals)
    
    print('Unique students after adding counterfactuals:', df_students_cf['student_id_scram'].nunique())
    
    # add in transit for each school in portfolio
    df_students_ext = add_transit_info(df_students_cf, df_programs, df_zipzip)
    
    print('Unique students after adding transit info:', df_students_ext['student_id_scram'].nunique())
    
    # add in quality metrics for each school in portfolio
    df_students_ext = add_quality_info(df_students_ext, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='offer_rate')
    df_students_ext = add_quality_info(df_students_ext, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='impact')
    df_students_ext = add_quality_info(df_students_ext, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='performance')
    df_students_ext = add_quality_info(df_students_ext, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='program_grad_rate')
    df_students_ext = add_quality_info(df_students_ext, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='program_college_rate')
    df_students_ext = add_quality_info(df_students_ext, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='mean_SQR')
    df_students_ext = add_quality_info(df_students_ext, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='aggregated_quality')
   
    
    print('Unique students after adding portfolio metrics:', df_students_ext['student_id_scram'].nunique())
    
    # add cluster numbers
    ## Parameters ##
    k = 12 # number of choices
    n = 9 # number of clusters
    sort = True # whether to sort portfolios
    
    df_students_ext['offer_rates'] = df_students_ext.apply(lambda row: [row[f'offer_rate_{i}'] for i in range(1, 13) if pd.notnull(row[f'offer_rate_{i}'])], axis=1)
    df_students_ext['offer_rates_imputed'] = df_students_ext.apply(lambda row: [row[f'offer_rate_{i}'] if pd.notnull(row[f'offer_rate_{i}']) else 1 for i in range(1, 13)], axis=1)
    
    if impute_clusters:  # pad out to length 12 with offer_rate = 1
        df_students_clusters = df_students_ext
        df_students_clusters['offer_rates_imputed'] = df_students_clusters['offer_rates_imputed'].apply(lambda x: np.array(x))

        X = np.vstack(df_students_clusters['offer_rates_imputed'].to_numpy())
        X = np.sort(X, axis=1)

        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(X)

        label_counts = Counter(kmeans.labels_)
        max_count = np.max(list(label_counts.values()))

        df_students_clusters['cluster'] = kmeans.labels_
    
    else:  # only consider full-length lists
        df_students_clusters = df_students_ext[df_students_ext['offer_rates'].apply(lambda x: len(x) == k)]
        df_students_clusters['offer_rates'] = df_students_clusters['offer_rates'].apply(lambda x: np.array(x))

        X = np.vstack(df_students_clusters['offer_rates'].to_numpy())
        X = np.sort(X, axis=1)

        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(X)

        label_counts = Counter(kmeans.labels_)
        max_count = np.max(list(label_counts.values()))

        df_students_clusters['cluster'] = kmeans.labels_
        
    print('Unique students after adding clusters:', df_students_clusters['student_id_scram'].nunique())
    
    return df_students_clusters


def add_counterfactuals(df_students, df_counterfactuals):
    # Add in counterfactual match information to compute counterfactual metrics
    return df_students.join(df_counterfactuals, on='student_id_scram', how='left')


'''Given student df containing counterfactuals and program df containing metrics, add portfolio info w.r.t. metric'''
def add_quality_info(df_students_cf, df_programs, programs_counterfactual, df_zipzip, drop_helpers, metric='offer_rate'):
    metric_dict = dict(zip(df_programs['programcode'], df_programs[metric]))
    
    lower_is_better = (metric == 'offer_rate')   # for all other metrics, higher is better        
    
    # set df to be the students df
    df = df_students_cf

    # Add column that gives the metric for each ranked program, match, and final program for each student
    # for nan programs, impute offer rate 1 (but keep nan for all other metrics)
    if metric == 'offer_rate': nanvalue = 1
    else: nanvalue = np.nan
    
    for i in range(1, 13):
        df[f'{metric}_{i}'] = df[f'programcode{i}'].apply(lambda x: metric_dict[x] if pd.notnull(x) else nanvalue)
        df[f'{metric}_mp'] = df['mp_programcode'].apply(lambda x: metric_dict[x] if pd.notnull(x) else nanvalue)
            
    df[f'{metric}_match'] = np.where(df['matched'] == 1, df['matched_program'].map(metric_dict), nanvalue)
    df[f'{metric}_final'] = np.where(df['finalprogramcode'].notna(), df['finalprogramcode'].map(metric_dict), nanvalue)          

    
    # Add helper column that is equal to the metric if a student is admitted counterfactually and appropriately-imputed-value otherwise
    if metric == 'offer_rate': # impute 1 for offer rate
        for programcode in programs_counterfactual:
            df[f'{programcode}_helper'] = df[programcode].apply(lambda x: metric_dict[programcode] if x else 1.0)
    else:  # impute mp/final value for all other metrics
        for programcode in programs_counterfactual:
            df[f'{programcode}_helper'] = np.where(df[programcode], metric_dict[programcode], df[f'{metric}_final'])
        
    # Get the best metric a student could have matched to counterfactually, including the program the student actually matched to
    if metric == 'offer_rate':
        df[f'{metric}_best_cf'] = df.apply(lambda row: np.nanmin(row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in programs_counterfactual]]]), axis=1)
    else:
        df[f'{metric}_best_cf'] = df.apply(lambda row: np.nanmax(row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in programs_counterfactual]]]), axis=1)

    # Get a df that has zipcodes and programs as columns and the distance between the zipcode and the program
    zip_to_programs = defaultdict(lambda: [])

    for index, row in df_programs.iterrows():
        zip_to_programs[row['zipcode']].append(row['programcode'])

    df_zipzip['programcode'] = df_zipzip['zip_dest'].apply(lambda x: zip_to_programs[x])
    df_zipzip = df_zipzip.explode('programcode', ignore_index=True)

    # Get programs that are (not) majority Asian and white (according to previous year's match outcome)
    cohort_directory = f'R:/CR4239/Cohort 2021-22/'
    df_2021 = pd.read_csv(cohort_directory+f'2021 HSAPS_Scrambled_for2939.csv', dtype={'student_id_scram':'string'}, low_memory=False)
    df_2021 = df_2021.drop_duplicates(subset = 'student_id_scram')
    df_2021['asian_white'] = df_2021['ethnicity'].apply(lambda x: x == 'asian' or x == 'white')
    df_racial_composition = df_2021.groupby('finalprogramcode', as_index=False)['asian_white'].mean()
    majority_asian_white_programs = set(df_racial_composition[df_racial_composition['asian_white'] >= 0.5]['finalprogramcode'].to_list())
    not_majority_asian_white_programs = set(df_racial_composition[df_racial_composition['asian_white'] < 0.5]['finalprogramcode'].to_list())

    safe_programs = set(df_programs[df_programs['safe']]['programcode'].to_list())
    
    # Returns the set of programs within a certain radius (in transit minutes)
    # Set radius='match' to filter programs that are at least as close as match
    def filter_programs_by_radius(student_row, radius=30):
        if radius=='match':
            radius = student_row['transit_match']
            if pd.isnull(radius):
                radius = 30
        helper_df = df_zipzip[df_zipzip['zip_source']==student_row['zipcode']]
        programs = helper_df.loc[helper_df['commute_time'] <= radius, 'programcode'].to_list()
        return programs
    
    def best_cfs(row, lower_is_better):
        close_programs = filter_programs_by_radius(row, 'match')
        portfolio_programs = [row[f'programcode{i}'] for i in range(1, 13) if pd.notnull(row[f'programcode{i}'])]
        close_portfolio_programs = list(set(close_programs) & set(portfolio_programs))
        
        programs = list(set(close_programs) & set(programs_counterfactual))
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in programs]]].to_list()
        try:
            if lower_is_better: 
                best_cf_pareto = np.nanmin(metrics)  
                arg = np.nanargmin(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
            else: 
                best_cf_pareto = np.nanmax(metrics)
                arg = np.nanargmax(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
        except:
            return None
        
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in set(close_portfolio_programs) & set(programs_counterfactual)]]].to_list()
        if lower_is_better: 
            best_cf_pareto_withinportfolio = np.nanmin(metrics)
        else: 
            best_cf_pareto_withinportfolio = np.nanmax(metrics)
        
        return best_cf_pareto, best_cf_pareto_withinportfolio, best_cf_pareto_programcode

    def best_cfs_within_radius(row, lower_is_better, radius):
        close_programs = filter_programs_by_radius(row, radius)
        portfolio_programs = [row[f'programcode{i}'] for i in range(1, 13) if pd.notnull(row[f'programcode{i}'])]
        close_portfolio_programs = list(set(close_programs) & set(portfolio_programs))
        
        programs = list(set(close_programs) & set(programs_counterfactual))
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in programs]]].to_list()
        try:
            if lower_is_better: 
                best_cf_pareto = np.nanmin(metrics)  
                arg = np.nanargmin(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
            else: 
                best_cf_pareto = np.nanmax(metrics)
                arg = np.nanargmax(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
        except:
            return None
        
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in set(close_portfolio_programs) & set(programs_counterfactual)]]].to_list()
        if lower_is_better: 
            best_cf_pareto_withinportfolio = np.nanmin(metrics)
        else: 
            best_cf_pareto_withinportfolio = np.nanmax(metrics)
        
        return best_cf_pareto, best_cf_pareto_withinportfolio, best_cf_pareto_programcode

    def best_cfs_with_homophily(row, lower_is_better):
        close_programs = filter_programs_by_radius(row, 'match')
        portfolio_programs = [row[f'programcode{i}'] for i in range(1, 13) if pd.notnull(row[f'programcode{i}'])]
        close_portfolio_programs = list(set(close_programs) & set(portfolio_programs))

        if row['ethnicity'] == 'asian' or row['ethnicity'] == 'white':
            close_programs = list(set(close_programs) & majority_asian_white_programs)
            portfolio_programs = list(set(portfolio_programs) & majority_asian_white_programs)
            close_portfolio_programs = list(set(close_portfolio_programs) & majority_asian_white_programs)
        else:
            close_programs = list(set(close_programs) & not_majority_asian_white_programs)
            portfolio_programs = list(set(portfolio_programs) & not_majority_asian_white_programs)
            close_portfolio_programs = list(set(close_portfolio_programs) & not_majority_asian_white_programs)
        
        programs = list(set(close_programs) & set(programs_counterfactual))
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in programs]]].to_list()
        try:
            if lower_is_better: 
                best_cf_pareto = np.nanmin(metrics)  
                arg = np.nanargmin(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
            else: 
                best_cf_pareto = np.nanmax(metrics)
                arg = np.nanargmax(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
        except:
            return None
        
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in set(close_portfolio_programs) & set(programs_counterfactual)]]].to_list()
        if lower_is_better: 
            best_cf_pareto_withinportfolio = np.nanmin(metrics)
        else: 
            best_cf_pareto_withinportfolio = np.nanmax(metrics)
        
        return best_cf_pareto, best_cf_pareto_withinportfolio, best_cf_pareto_programcode

    def best_cfs_with_safety(row, lower_is_better):
        close_programs = filter_programs_by_radius(row, 'match')
        portfolio_programs = [row[f'programcode{i}'] for i in range(1, 13) if pd.notnull(row[f'programcode{i}'])]
        close_portfolio_programs = list(set(close_programs) & set(portfolio_programs))

        close_programs = list(set(close_programs) & safe_programs)
        portfolio_programs = list(set(portfolio_programs) & safe_programs)
        close_portfolio_programs = list(set(close_portfolio_programs) & safe_programs)
    
        programs = list(set(close_programs) & set(programs_counterfactual))
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in programs]]].to_list()
        try:
            if lower_is_better: 
                best_cf_pareto = np.nanmin(metrics)  
                arg = np.nanargmin(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
            else: 
                best_cf_pareto = np.nanmax(metrics)
                arg = np.nanargmax(metrics)
                if arg == 0:
                    best_cf_pareto_programcode = row['matched_program']
                else:
                    best_cf_pareto_programcode = programs[arg - 1]
        except:
            return None
        
        metrics = row[[f'{metric}_match', *[f'{programcode}_helper' for programcode in set(close_portfolio_programs) & set(programs_counterfactual)]]].to_list()
        if lower_is_better: 
            best_cf_pareto_withinportfolio = np.nanmin(metrics)
        else: 
            best_cf_pareto_withinportfolio = np.nanmax(metrics)
        
        return best_cf_pareto, best_cf_pareto_withinportfolio, best_cf_pareto_programcode
    
    df[[f'{metric}_best_cf_pareto', f'{metric}_best_cf_pareto_withinportfolio', f'{metric}_best_cf_pareto_programcode']] = df.apply(lambda row: best_cfs(row, lower_is_better), axis=1, result_type='expand')

    df[[f'{metric}_best_cf_pareto_homophily', f'{metric}_best_cf_pareto_withinportfolio_homophily', f'{metric}_best_cf_pareto_programcode_homophily']] = df.apply(lambda row: best_cfs_with_homophily(row, lower_is_better), axis=1, result_type='expand')

    df[[f'{metric}_best_cf_pareto_safe', f'{metric}_best_cf_pareto_withinportfolio_safe', f'{metric}_best_cf_pareto_programcode_safe']] = df.apply(lambda row: best_cfs_with_safety(row, lower_is_better), axis=1, result_type='expand')

    df[[f'{metric}_best_cf_pareto_within_33', f'{metric}_best_cf_pareto_withinportfolio_within_33', f'{metric}_best_cf_pareto_programcode_within_33']] = df.apply(lambda row: best_cfs_within_radius(row, lower_is_better, 33), axis=1, result_type='expand')

    
    # drop helper columns
    if drop_helpers:
        df.drop(columns=[col for col in df.columns if 'helper' in col], inplace=True)        
    
    return df


'''Given student df, program df, and df_zipzip, add transits for each school in portfolio'''
def add_transit_info(df_students, df_programs, df_zipzip):
    # set df to be the students df
    df = df_students
    df = df.drop_duplicates(subset = 'student_id_scram')

    # Get a df that has zipcodes and programs as columns and the distance between the zipcode and the program
    zip_to_programs = defaultdict(lambda: [])

    for index, row in df_programs.iterrows():
        zip_to_programs[row['zipcode']].append(row['programcode'])

    df_zipzip['programcode'] = df_zipzip['zip_dest'].apply(lambda x: zip_to_programs[x])
    df_zipzip = df_zipzip.explode('programcode', ignore_index=True)
    df_zipzip = df_zipzip.dropna(subset = ['zip_source','programcode'])
    print(df_zipzip.duplicated(subset = ['zip_source','programcode']).any())
    
    # replace NaN with a sentinel value before merge
    progcode_cols = [f'programcode{i}' for i in range(1,13)] + ['matched_program']
    
    # Add column that gives the transit for each ranked program for each student
    for i in range(1, 13):
        mergedf = pd.merge(df[['student_id_scram','zipcode',f'programcode{i}']], df_zipzip, how='left', 
                                      left_on=['zipcode',f'programcode{i}'], right_on=['zip_source','programcode'])
        
        mergedf = mergedf.rename(columns = {'commute_time': f'transit_{i}'})
        if f'transit_{i}' in df.columns:
            del df[f'transit_{i}']

        df = pd.merge(df, mergedf[['student_id_scram', f'transit_{i}']], how = 'left', on = 'student_id_scram')


    # transit for matched program
    mergedf = pd.merge(df[['student_id_scram','zipcode','matched_program']], df_zipzip, how='left', 
                                      left_on=['zipcode','matched_program'], right_on=['zip_source','programcode'])
        
    mergedf = mergedf.rename(columns = {'commute_time': 'transit_match'})
    if 'transit_match' in df.columns:
        del df['transit_match']

    df = pd.merge(df, mergedf[['student_id_scram', 'transit_match']], how = 'left', on = 'student_id_scram')
    
    return df