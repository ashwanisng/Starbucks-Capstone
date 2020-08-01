import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_portfolio(portfolio):
    
    portfolio_clean = portfolio.copy()
    
    d_chann = pd.get_dummies(portfolio_clean.channels.apply(pd.Series).stack(),
                             prefix="channel").sum(level=0)
    
    portfolio_clean = pd.concat([portfolio_clean, d_chann], axis=1, sort=False)
    portfolio_clean.drop(columns='channels', inplace=True)
    
    portfolio_clean.rename(columns={'id':'offer_id'}, inplace=True)

    return portfolio_clean


def clean_profile(profile):
    
    
    profile_clean = profile.copy()
    
    date = lambda x: pd.to_datetime(str(x), format='%Y%m%d')
    
    profile_clean.became_member_on = profile_clean.became_member_on.apply(date)
    
    profile_clean['valid'] = (profile_clean.age != 118).astype(int)

    profile_clean.rename(columns={'id':'customer_id'}, inplace=True)

    dummy_gender = pd.get_dummies(profile_clean.gender, prefix="gender")
    
    profile_clean = pd.concat([profile_clean, dummy_gender], axis=1, sort=False)
    
    return profile_clean


def clean_transcript(transcript):
   
    transcript_clean = transcript.copy()

    transcript_clean.event = transcript_clean.event.str.replace(' ', '_')
    
    dummy_event = pd.get_dummies(transcript_clean.event, prefix="event")
    
    transcript_clean = pd.concat([transcript_clean, dummy_event], axis=1,
                                 sort=False)
    transcript_clean.drop(columns='event', inplace=True)

    transcript_clean['offer_id'] = [[*v.values()][0]
                                    if [*v.keys()][0] in ['offer id',
                                                          'offer_id'] else None
                                    for v in transcript_clean.value]

    
    transcript_clean['amount'] = [np.round([*v.values()][0], decimals=2)
                                  if [*v.keys()][0] == 'amount' else None
                                  for v in transcript_clean.value]
    transcript_clean.drop(columns='value', inplace=True)

    transcript_clean.rename(columns={'person':'customer_id'}, inplace=True)
    
    return transcript_clean

def merge_datasets(portfolio_clean, profile_clean, transcript_clean):
  
    trans_prof = pd.merge(transcript_clean, profile_clean, on='customer_id',
                          how="left")
    df = pd.merge(trans_prof, portfolio_clean, on='offer_id', how='left')

    offer_id = {'ae264e3637204a6fb9bb56bc8210ddfd': 'B1',
                '4d5c57ea9a6940dd891ad53e9dbe8da0': 'B2',
                '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'B3',
                'f19421c1d4aa40978ebb69ca19b0e20d': 'B4',
                '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'D1',
                '2298d6c36e964ae4a3e7e9706d1fb8c2': 'D2',
                'fafdcd668e3743c1bb461111dcafc2a4': 'D3',
                '2906b810c7d4411798c6938adc9daaa5': 'D4',
                '3f207df678b143eea3cee63160fa8bed': 'I1',
                '5a8bc65990b245e5a138643cd4eb9837': 'I2'}
    df.offer_id = df.offer_id.apply(lambda x: offer_id[x] if x else None)

    return df


def get_offer_cust(df, offer_type=None):
    
    data = dict()
    for e in ['received', 'viewed', 'completed']:

        if offer_type == 'informational' and e == 'completed':
            continue
        flag = (df['event_offer_{}'.format(e)] == 1)
        
        key = e
        
        if offer_type:
            flag = flag & (df.offer_type == offer_type)
            
            key = '{}_'.format(offer_type) + key
            
        data[key] = df[flag].groupby('customer_id').offer_id.count()
        

        flag = (df.event_offer_completed == 1)
        
    if offer_type != 'informational':
        key = 'reward'
        if offer_type:
            
            flag = flag & (df.offer_type == offer_type)
            
            key = '{}_'.format(offer_type) + key
            
        data[key] = df[flag].groupby('customer_id').reward.sum()

    return data


def get_offer_id_cust(df, offer_id):
 
    data = dict()

    for e in ['received', 'viewed', 'completed']:

        if offer_id in ['I1', 'I2'] and e == 'completed':
            continue
        event = 'event_offer_{}'.format(e)
        
        flag = (df[event] == 1) & (df.offer_id == offer_id)
        
        key = '{}_{}'.format(offer_id, e)
        
        data[key] = df[flag].groupby('customer_id').offer_id.count()

    flag = (df.event_offer_completed == 1) & (df.offer_id == offer_id)
    
    if offer_id not in ['I1', 'I2']:
        
        key = '{}_reward'.format(offer_id)
        
        data[key] = df[flag].groupby('customer_id').reward.sum()

    return data


def round_age(x):
    
    for y in range(15, 106, 10):
        
        if x >= y and x < y+10:
            
            return y
    return 0


def round_income(x):
  
    for y in range(30, 130, 10):
        
        if x >= y*1000 and x < (y+10)*1000:
            
            return y*1000
    return 0


def per_customer_data(df, profile):
    
   
    cust_dict = dict()
    

    transactions = df[df.event_transaction == 1].groupby('customer_id')
    
    
    cust_dict['total_expense'] = transactions.amount.sum()
    
    cust_dict['total_transactions'] = transactions.amount.count()
    
    cust_dict.update(get_offer_cust(df))

    for ot in ['bogo', 'discount', 'informational']:
        
        cust_dict.update(get_offer_cust(df, ot))
        
        for oi in ['B1', 'B2', 'B3', 'B4', 'D1', 'D2', 'D3', 'D4', 'I1', 'I2']:
            
            cust_dict.update(get_offer_id_cust(df, oi))

    customers = pd.concat(cust_dict.values(), axis=1, sort=False);
    customers.columns = cust_dict.keys()
    customers.fillna(0, inplace=True)

    customers = pd.merge(customers, profile.set_index('customer_id'),
                         left_index=True, right_index=True)
    customers['age_group'] = customers.age.apply(round_age)
    customers['income_group'] = customers.income.apply(round_income)
    customers['net_expense'] = customers['total_expense'] - customers['reward']

    return customers


def get_offer_stat(customers, stat, offer):
   
    valid = (customers.valid == 1)
    
    rcv_col = '{}_received'.format(offer)
    
    vwd_col = '{}_viewed'.format(offer)
    
    received = valid & (customers[rcv_col] > 0) & (customers[vwd_col] == 0)
    cpd = None
    
    if offer not in ['informational', 'I1', 'I2']:
        cpd_col = '{}_completed'.format(offer)
        
        viewed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] == 0)
        
        completed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] > 0)
        cpd = customers[completed][stat]
    else:
        viewed = valid & (customers[vwd_col] > 0)

    return customers[received][stat], customers[viewed][stat], cpd


def get_average_expense(customers, offer):
   
    rcv_total, vwd_total, cpd_total = get_offer_stat(customers,
                                                     'total_expense', offer)
    rcv_trans, vwd_trans, cpd_trans = get_offer_stat(customers,
                                                     'total_transactions',
                                                     offer)

    rcv_avg = rcv_total / rcv_trans
    
    rcv_avg.fillna(0, inplace=True)
    
    vwd_avg = vwd_total / vwd_trans
    vwd_avg.fillna(0, inplace=True)
    

    cpd_avg = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_avg = cpd_total / cpd_trans

    return rcv_avg, vwd_avg, cpd_avg


def get_average_reward(customers, offer):
   
    cpd_col = '{}_completed'.format(offer)
    
    rwd_col = '{}_reward'.format(offer)
    
    completed = customers[(customers.valid == 1) & (customers[cpd_col] > 0)]

    return completed[rwd_col] / completed[cpd_col]


def get_offer_stat_by(customers, stat, offer, by_col, aggr='sum'):

    valid = (customers.valid == 1)
    
    rcv_col = '{}_received'.format(offer)
    
    vwd_col = '{}_viewed'.format(offer)
    
    received = valid & (customers[rcv_col] > 0) & (customers[vwd_col] == 0)
    cpd = None
    
    if offer not in ['informational', 'I1', 'I2']:
        
        cpd_col = '{}_completed'.format(offer)
        
        viewed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] == 0)
        
        completed = valid & (customers[cpd_col] > 0)
        
        
        if aggr == 'sum':
            cpd = customers[completed].groupby(by_col)[stat].sum()
            
        elif aggr == 'mean':
            cpd = customers[completed].groupby(by_col)[stat].mean()
    else:
        viewed = valid & (customers[vwd_col] > 0)
        
    if aggr == 'sum':
        rcv = customers[received].groupby(by_col)[stat].sum()
        vwd = customers[viewed].groupby(by_col)[stat].sum()
        
    elif aggr == 'mean':
        rcv = customers[received].groupby(by_col)[stat].mean()
        vwd = customers[viewed].groupby(by_col)[stat].mean()

    return rcv, vwd, cpd


def get_average_expense_by(customers, offer, by_col):
 
    rcv_total, vwd_total, cpd_total = get_offer_stat_by(customers,
                                                        'total_expense',
                                                        offer, by_col)
    rcv_trans, vwd_trans, cpd_trans = get_offer_stat_by(customers,
                                                        'total_transactions',
                                                        offer, by_col)

    rcv_avg = rcv_total / rcv_trans
    
    rcv_avg.fillna(0, inplace=True)
    
    vwd_avg = vwd_total / vwd_trans
    
    vwd_avg.fillna(0, inplace=True)

    cpd_avg = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_avg = cpd_total / cpd_trans

    return rcv_avg, vwd_avg, cpd_avg


def get_average_reward_by(customers, offer, by_col):
 
    cpd_col = '{}_completed'.format(offer)
    
    rwd_col = '{}_reward'.format(offer)
    
    completed = customers[(customers.valid == 1) &
                          (customers[cpd_col] > 0)].groupby(by_col)

    return completed[rwd_col].sum() / completed[cpd_col].count()



def plot_offer_expense(customers, offer):
 
    rcv, vwd, cpd = get_offer_stat(customers, 'total_expense', offer)
    
    rcv_avg, vwd_avg, cpd_avg = get_average_expense(customers, offer)

    plt.figure(figsize=(16, 5))
    bins = 100

    plt.subplot(121)
    plt.hist(rcv, bins, alpha=0.5, label='{}-received'.format(offer))
    plt.hist(vwd, bins, alpha=0.5, label='{}-viewed'.format(offer))
    
    if offer not in ['informational', 'I1', 'I2']:
        plt.hist(cpd, bins, alpha=0.5, label='{}-completed'.format(offer))
    plt.legend(loc='best')
    
    ax = plt.gca();
    ax.set_xlim(0, 600);
    plt.title('Total Transaction ($)')
    plt.grid();

    plt.subplot(122)
    plt.hist(rcv_avg, bins, alpha=0.5, label='{}-received'.format(offer))
    
    plt.hist(vwd_avg, bins, alpha=0.5, label='{}-viewed'.format(offer))
    
    if offer not in ['informational', 'I1', 'I2']:
        plt.hist(cpd_avg, bins, alpha=0.5, label='{}-completed'.format(offer))
    plt.legend(loc='best')
    ax = plt.gca();
    ax.set_xlim(0, 50);
    plt.title('Average Transaction ($)')
    plt.grid();


def plot_offer_reward(customers, offer):
   
    plt.figure(figsize=(16, 5))
    bins = 10

    key = '{}_completed'.format(offer)
    
    key_avg = '{}_reward'.format(offer)
    
    rwd = customers[(customers.valid == 1) & (customers[key] > 0)][key_avg]
    
    rwd_avg = get_average_reward(customers, offer)

    plt.subplot(121)
    plt.hist(rwd, bins, alpha=0.5, label=offer)
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.grid();

    plt.subplot(122)
    plt.hist(rwd_avg, bins, alpha=0.5, label=offer)
    plt.title('Average Reward ($)');
    plt.legend(loc='best');
    plt.grid();


def plot_offer_expense_by(customers, offer):
   
    rcv_by = dict()
    vwd_by = dict()
    cpd_by = dict()
    rcv_avg_by = dict()
    vwd_avg_by = dict()
    cpd_avg_by = dict()

    for key in ['age_group', 'income_group', 'gender']:
        rcv_by[key], vwd_by[key], cpd_by[key] = get_offer_stat_by(customers,
                                                                  'net_expense',
                                                                  offer, key,
                                                                  aggr='mean')
        by_data = get_average_expense_by(customers, offer, key)
        rcv_avg_by[key], vwd_avg_by[key], cpd_avg_by[key] = by_data

    plt.figure(figsize=(16, 10))

    plt.subplot(231)
    plt.plot(rcv_by['age_group'], label='{}-received'.format(offer))
    
    plt.plot(vwd_by['age_group'], label='{}-viewed'.format(offer))
    
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_by['age_group'], label='{}-completed'.format(offer))
        
    plt.legend(loc='best')
    plt.title('Net Expense');
    plt.grid();

    plt.subplot(232)
    plt.plot(rcv_by['income_group'], label='{}-received'.format(offer))
    plt.plot(vwd_by['income_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_by['income_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Net Expense');
    plt.grid();

    index = np.array([0, 1, 2])
    bar_width = 0.3
    plt.subplot(233)
    plt.bar(index, rcv_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-received'.format(offer))
    plt.bar(index + bar_width, vwd_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.bar(index + 2*bar_width, cpd_by['gender'].reindex(['M', 'F', 'O']),
                bar_width, label='{}-completed'.format(offer))
    plt.grid();
    plt.legend(loc='best');
    plt.title('Net Expense');
    plt.xticks(index + bar_width, ('M', 'F', 'O'));

    plt.subplot(234)
    plt.plot(rcv_avg_by['age_group'], label='{}-received'.format(offer))
    plt.plot(vwd_avg_by['age_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_avg_by['age_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Average Transaction Value');
    plt.grid();

    plt.subplot(235)
    plt.plot(rcv_avg_by['income_group'], label='{}-received'.format(offer))
    plt.plot(vwd_avg_by['income_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_avg_by['income_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Average Transaction Value');
    plt.grid();

    plt.subplot(236)
    plt.bar(index, rcv_avg_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label='{}-received'.format(offer))
    plt.bar(index + bar_width, vwd_avg_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.bar(index+2*bar_width, cpd_avg_by['gender'].reindex(['M', 'F', 'O']),
                bar_width, label='{}-completed'.format(offer))
    plt.grid();
    plt.legend(loc='best');
    plt.title('Average Transaction Value');
    plt.xticks(index + bar_width, ('M', 'F', 'O'));


def plot_offer_reward_by(customers, offer):
    
    rwd_by = dict()
    rwd_avg_by = dict()

    for key in ['age_group', 'income_group', 'gender']:
        key_cpd = '{}_completed'.format(offer)
        
        key_rwd = '{}_reward'.format(offer)
        
        offer_cpd = customers[(customers.valid == 1) &
                              (customers[key_cpd] > 0)].groupby(key)
        rwd_by[key] = offer_cpd[key_rwd].mean()
        rwd_avg_by[key] = get_average_reward_by(customers, offer, key)

    plt.figure(figsize=(16, 10))

    plt.subplot(231)
    plt.plot(rwd_by['age_group'], label=offer)
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.grid();

    plt.subplot(232)
    plt.plot(rwd_by['income_group'], label=offer)
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.grid();

    index = np.array([0, 1, 2])
    bar_width = 0.3
    plt.subplot(233)
    plt.bar(index, rwd_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label=offer)
    plt.grid();
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.xticks(index, ('M', 'F', 'O'));

    plt.subplot(234)
    plt.plot(rwd_avg_by['age_group'], label=offer)
    plt.title('Average Reward');
    plt.legend(loc='best');
    plt.grid();
    ax = plt.gca();
    ymax = rwd_avg_by['age_group'].max() + 1
    ax.set_ylim(0, ymax);

    plt.subplot(235)
    plt.plot(rwd_avg_by['income_group'], label=offer)
    plt.title('Average Reward');
    plt.legend(loc='best');
    plt.grid();
    ax = plt.gca();
    ymax = rwd_avg_by['income_group'].max() + 1
    ax.set_ylim(0, ymax);

    plt.subplot(236)
    plt.bar(index, rwd_avg_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label=offer)
    plt.grid();
    plt.title('Average Reward');
    plt.legend(loc='best');
    plt.xticks(index, ('M', 'F', 'O'));


def get_net_expense(customers, offer, q=0.5):
    
    flag = (customers['{}_viewed'.format(offer)] > 0)
    
    flag = flag & (customers.net_expense > 0)
    
    flag = flag & (customers.total_transactions >= 5)
    
    if offer not in ['I1', 'I2']:
        
        flag = flag & (customers['{}_completed'.format(offer)] > 0)
        
    return customers[flag].net_expense.quantile(q)


def get_most_popular_offers(customers, n_top=2, q=0.5, offers=None):
    
    if not offers:
        offers = ['I1', 'I2', 'B1', 'B2', 'B3',
                  'B4', 'D1', 'D2', 'D3', 'D4']
    offers.sort(key=lambda x: get_net_expense(customers, x, q), reverse=True)
    offers_dict = {o: get_net_expense(customers, o, q) for o in offers}
    return offers[:n_top], offers_dict


def get_most_popular_offers_filtered(customers, n_top=2, q=0.5, income=None,
                                     age=None, gender=None):
  
    flag = (customers.valid == 1)
    
    if income:
        income_gr = round_income(income)
        
        if income_gr > 0:
            flag = flag & (customers.income_group == income_gr)
            
    if age:
        age_gr = round_age(age)
        
        if age_gr > 0:
            flag = flag & (customers.age_group == age_gr)
            
    if gender:
        flag = flag & (customers.gender == gender)
        
    return get_most_popular_offers(customers[flag], n_top, q)
