import matplotlib.pyplot as plt
import seaborn as sns

def create_bar_chart(df_students, metric, group='ethnicity', save=False, title=''
                     , savefolder= f'../plots_export/', box = True, do_poverty = True):
    
    group_names_pretty = {
        'asian': 'Asian', 'black':'Black', 'hispanic':'Hispanic', 'white':'white',
        'M':'Male',
        'F' : 'Female'
    }
    
    dftoplot = df_students[df_students[group].isin(group_names_pretty.keys())].copy()
    dftoplot[group].replace(to_replace = group_names_pretty, inplace = True)  
    
    # alphabetical ordering of groups
    dftoplot = dftoplot.sort_values(by = group)

    if do_poverty:
        dopovertylist = [False, True]
    else:
        dopovertylist = [False]
    
    for do_poverty_local in dopovertylist:
        if do_poverty_local:
            dftoplot = dftoplot.rename(columns = {'poverty':'FRL'})
            dftoplot['FRL'].replace(to_replace = {0 : "False", 1: "True"}, inplace = True)
            ax = sns.barplot(data = dftoplot, y = metric, x = group, hue = 'FRL', hue_order = ['False', 'True'])
            plt.legend(title = 'Free/Reduced Lunch', ncol = 2, loc = 'upper left', bbox_to_anchor=(0, 1.07))
            
        else:
            sns.barplot(data = dftoplot, y = metric, x = group, color = '#1f77b4')
        plt.xlabel(None)

        print(dftoplot.groupby(group)[metric].mean())

        if len(title) > 0:
            plt.ylabel(title, fontsize=12)
        else:
            plt.ylabel(metric, fontsize=12)
            
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        sns.despine()
        plt.tight_layout()

        if save and do_poverty_local:
            plt.savefig(f'{savefolder}/{metric}_{group}_poverty.pdf', dpi=600)
        else:
            plt.savefig(f'{savefolder}/{metric}_{group}.pdf', dpi=600)

        plt.show()
        

def create_stacked_bar_chart(df_students, metrics, labels, group='ethnicity', save=False, title=''
                     , savefolder= f'../plots_export/', box = True, do_poverty = False):
    
    group_names_pretty = {
        'asian': 'Asian', 'black':'Black', 'hispanic':'Hispanic', 'white':'white',
        'M':'Male',
        'F' : 'Female'
    }
    
    dftoplot = df_students[df_students[group].isin(group_names_pretty.keys())]
    dftoplot[group].replace(to_replace = group_names_pretty, inplace = True)
    
    
    # alphabetical ordering of groups
    dftoplot = dftoplot.sort_values(by = group)
    
    colors = ['#add8e6', '#5f9ea0', '#1f77b4'] #, '#00008b']

    if do_poverty:
        for metric in metrics:
            sns.barplot(data = dftoplot, y = metric, x = group, hue = 'poverty')

    else:
        for i, metric in enumerate(metrics):
            sns.barplot(data = dftoplot, y = metric, x = group, color = colors[i], label = labels[i])
            print(dftoplot.groupby(group)[metric].mean())
    plt.xlabel(None)
    

    if len(title) > 0:
        plt.ylabel(title, fontsize=12)
    else:
        plt.ylabel(metric, fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    sns.despine()
    plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5,1.1), ncols=3)
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{savefolder}/{title}_{group}_stacked.pdf', dpi=600)
    
    plt.show()
    

def plot_portfolio_outcome(row, title='portfolio', save=False, savefolder= f'../plots_export/'):
    offer_rates = row['offer_rates'].to_list()[0]
    if row['matched'].to_list()[0] == 1:
        match_choice_num = int(row['matched_choice_num'].to_list()[0])
    else:
        match_choice_num = len(offer_rates) + 1
    print(match_choice_num)
    plt.figure(figsize=(6, 2))
    plt.scatter(range(1, match_choice_num), offer_rates[:match_choice_num - 1], marker='x', color='red', label='reject')
    if match_choice_num <= len(offer_rates):
        plt.scatter([match_choice_num], offer_rates[match_choice_num - 1], color='green', label='match')
    if match_choice_num < len(offer_rates):
        plt.scatter(range(match_choice_num + 1, len(offer_rates) + 1), offer_rates[match_choice_num:], color='black', alpha=0.2)
    plt.xlim(0.5, 12.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('rank')
    plt.ylabel('offer rate')
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(f'{savefolder}/{title}.pdf', dpi=600)
    plt.show()
    
def plot_portfolio_recommendation(row, title='portfolio', save=False, savefolder= f'../plots_export/'):
    offer_rates = row['offer_rates'].to_list()[0]
    best_arg, best_move = row['best_move']
    plt.figure(figsize=(6, 2))
    plt.scatter(range(1, len(offer_rates)+1), offer_rates, color='black')
    plt.scatter([best_arg + 1], [offer_rates[best_arg]], marker='x', color='blue', label='old choice')
    plt.scatter([best_arg + 1], [offer_rates[best_arg] + best_move], marker='x', color='red', label='recommended choice')
    plt.xlim(0.5, 12.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('rank')
    plt.ylabel('offer rate')
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(f'{savefolder}/{title}.pdf', dpi=600)
    plt.show()
    