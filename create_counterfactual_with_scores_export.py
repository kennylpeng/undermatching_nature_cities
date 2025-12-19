import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


'''Returns students_df updated with columns:
{programcode}: 1 if counterfactual offer 0 if not,
selectivity_match: selectivity rate of matched program,
selectivity_avg: average selectivity rate of listed programs,
selectivity_{i} for i in [12]: selectivity rate of i-th ranked program in portfolio,
selectivity_best_cf: selectivity rate of 1st percentile most selective program a student received a counterfactual offer from,
selectivity_best_cf_inportfolio: selectivity rate of most selective program in a student's portfolio that the student received a counterfactual offer from,
regret: selectivity_match - selectivity_best_cf,
regret_inportfolio: selectivity_match - selectivity_best_cf_inportfolio

Also returns programs_counterfactual, a list of programs for which we compute counterfactual offers
'''
def get_seatgroup_scores(students_df, programs_df):

    # Load relevant student data
    
    def get_tiebreaker_num(tiebreaker):
        if type(tiebreaker)==str:
            head = tiebreaker[:5]
            return int(format(int(head, 16), 'd'))/1048518
        else:
            return np.nan
    
    def get_borough_code(dbn):
        if isinstance(dbn, str):
            return dbn[2]
        else:
            return np.nan

    students_df = students_df.set_index('student_id_scram')
    students_df['tiebreaker_num'] = students_df['tiebreaker'].apply(lambda dbn: get_tiebreaker_num(dbn))
    students_df['currentschoolborough']=students_df['currentschooldbn'].apply(get_borough_code)
    students_df['tier'] = students_df['avg_grades_for_tier'].apply(
        lambda x: 400 if x >= 350 else (300 if x >= 250 else (200 if x >= 150 else 100))
    )

    # load relevant program data
    
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
    
    for index, row in programs_df.iterrows():
        program_dict[row['programcode']] = {
            'requirements': row['requirements'],
            'method': row['method'],
            'eligibility': row['eligibility'],
            'audition_info': row['audition_info'],
            'priorities': row['priorities'],
            'grade_weight': row['grade_weight'],
            'dia_details': row['dia_details'],
            'schooldbn': row['dbn'],
            'dia_fraction': row['dia_fraction']
        }

    
    # Get pure screened programs (screened programs that only use grades)
    
    pure_screened_programs = []
    for programcode in program_dict.keys():
        if program_dict[programcode]['method']=='Screened':
            if program_dict[programcode]['grade_weight']==1:
                pure_screened_programs.append(programcode)


    # Helper functions to compute scores

    def get_continuing_col(students_df, programcode):
        schooldbn = program_dict[programcode]['schooldbn']
        continuing_col = students_df['currentschooldbn'].apply(lambda x: x == schooldbn)
        return continuing_col

    def get_borough_col(students_df, borough='K'):
        col1 = students_df['resborough'].apply(lambda x: x == borough)
        col2 = students_df['currentschoolborough'].apply(lambda x: x == borough)
        return col1 | col2

    def get_edopt_col(students_df, group='HIGH'):
        if group == 'HIGH':
            return students_df['edoptcategory'].apply(lambda x: 2 if x == 'HIGH' else (1 if x == 'MIDDLE' else 0))
        elif group == 'MIDDLE':
            return students_df['edoptcategory'].apply(lambda x: 2 if x == 'MIDDLE' else (1 if x == 'HIGH' else 0))
        elif group == 'LOW':
            return students_df['edoptcategory'].apply(lambda x: 2 if x == 'LOW' else (1 if x == 'HIGH' else 0))

    def get_dia_col(students_df):
        return students_df['poverty'].apply(lambda x: x)

    def get_priority_col(students_df, priority):
        if pd.isnull(priority):
            return 0
        if priority == 'Priority to continuing 8th graders':
            return get_continuing_col(students_df, programcode)
        if 'Brooklyn students or residents' in priority:
            return get_borough_col(students_df, 'K')
        if 'Bronx students or residents' in priority:
            return get_borough_col(students_df, 'X')
        if 'Manhattan students or residents' in priority:
            return get_borough_col(students_df, 'M')
        if 'Queens students or residents' in priority:
            return get_borough_col(students_df, 'Q')
        if 'Staten Island students or residents' in priority:
            return get_borough_col(students_df, 'R')
        return 0

    
    
    '''Returns a num. seat groups x num. students list of cols, giving the score of each student at each seat group'''
    def score(students_df, programcode):
        tiebreaker_col = 1 - students_df['tiebreaker_num'] # between 0 and 1
        priorities = program_dict[programcode]['priorities']

        priority_col = students_df['tiebreaker_num'].apply(lambda x: 0)
        for i, priority in enumerate(priorities):
            priority_col = priority_col + 100 * 10**(len(priorities)-i) * get_priority_col(students_df, priority) # 0 or >= 1000

        score_col = tiebreaker_col + priority_col

        if program_dict[programcode]['method']=='Screened':
            tier_col = students_df['tier'] # between 100 and 400
            score_col = score_col + tier_col

        score_cols = [score_col]

        if program_dict[programcode]['method']=='Ed. Opt.':
            ed_opt_high = get_edopt_col(students_df, group='HIGH')
            ed_opt_middle = get_edopt_col(students_df, group='MIDDLE')
            ed_opt_low = get_edopt_col(students_df, group='LOW')

            score_cols = [score_col + ed_opt_col for ed_opt_col in [ed_opt_high, ed_opt_middle, ed_opt_low]]

        if program_dict[programcode]['dia_fraction'] > 0:
            dia_score_cols = []
            for score_col in score_cols:
                dia_col = get_dia_col(students_df)
                dia_score_cols.append(score_col + dia_col)
                dia_score_cols.append(score_col)    
            score_cols = dia_score_cols

        return [score_col for score_col in score_cols]
    
    '''Returns an array of length num. seat groups, giving the proportion of seats allocated to each seat group'''
    def get_seat_group_fractions(programcode):
        seat_group_fractions = [1]
        if program_dict[programcode]['method']=='Ed. Opt.':
            seat_group_fractions = [0.4, 0.3, 0.3]
        dia_fraction = program_dict[programcode]['dia_fraction']

        if dia_fraction > 0:
            fractions = []
            for fraction in seat_group_fractions:
                fractions.append(fraction * dia_fraction)
                fractions.append(fraction * (1 - dia_fraction))
            seat_group_fractions = fractions

        return seat_group_fractions

    '''Returns an array of cutoffs corresponding to each seatgroup of a programcode.'''
    def cutoffs_seatgroups(programcode):
        seat_group_fractions = get_seat_group_fractions(programcode)

        offers, rejects = offer_reject_dict[programcode]['offers'], offer_reject_dict[programcode]['rejects']
        offered_scores = score(students_df.loc[offers], programcode)
        rejected_scores = score(students_df.loc[rejects], programcode)

        offered_scores = [x.to_list() for x in offered_scores]
        rejected_scores = [x.to_list() for x in rejected_scores]

        if len(offered_scores[0]) ==0 or len(rejected_scores[0]) == 0:
            return [0 for _ in seat_group_fractions]

        # only exists in case when there are some rejections, otherwise everyone is admitted
        total_seats = len(offered_scores[0])    
        cutoffs = []

        offered = np.array([])
        for i in range(len(seat_group_fractions)):
            seatgroup_scores = np.array(rejected_scores[i] + offered_scores[i])
            if len(offered) > 0:
                seatgroup_scores[offered] = -1 # if already offered, cannot match to seat group, so set to -1

            args_sorted = np.argsort(-np.array(seatgroup_scores))
            cutoffs.append(-np.sort(-seatgroup_scores)[int(total_seats * seat_group_fractions[i])-1])
            offered = np.concatenate((offered, args_sorted[:int(total_seats * seat_group_fractions[i])])).astype(int)

        return cutoffs

    # for each program in a subset of programs, compute the acceptance cutoff
    edopt_programs = [programcode for programcode in program_dict.keys() if program_dict[programcode]['method'] == 'Ed. Opt.']
    open_programs = [programcode for programcode in program_dict.keys() if program_dict[programcode]['method'] == 'Open']
    programs = list(set(edopt_programs) | set(open_programs) | set(pure_screened_programs))
    
    seatgroups = []
    seat_group_fractions_dict = {}
    for programcode in programs:
        # print(programcode)
        scores = score(students_df, programcode)
        seat_group_fractions_dict[programcode] = get_seat_group_fractions(programcode)
        for i in range(len(scores)):
            students_df[f'{programcode}-{i}'] = scores[i]
            seatgroups.append(f'{programcode}-{i}')

    df = students_df
    df = students_df[seatgroups]

    return df, seat_group_fractions_dict, programs