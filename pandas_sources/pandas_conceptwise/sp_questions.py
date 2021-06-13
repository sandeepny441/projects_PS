df_expl = df.explode('periods_to_remove').reset_index(drop=True)
df_final = pd.concat([df_expl, periods_to_remove_df], axis=1)
df_fcst_mau = df_fcst_mau.groupby('country').apply(lambda grp: grp.interpolate()
df_fcst_mau['incr_mau'].fillna(0, inplace=True)
df['A'].unique() / nunique()

df_fc_adj['quantity_adj'] = df_fc_adj['quantity'] - df_fc_adj['gap_adj']
column_c = column_b - column_a 
is_not_fom = [not d.is_month_start for d in df_incr_mau['date']]
#using not at the beginning 
iloc, loc
reset_index, drop=True 
drop columns, axis =1 is always mandatory?
concat vs append which is better?
free_df = pd.merge(df_paid_mau, df_fod_trial, on=join_cols, how='outer',suffixes=('_paid', '_ns'))
np.where 
df_adds['quantity'] = (df_adds['quantity'] - df_adds.groupby('country').shift(1)['quantity'])
pd.to_datetime()
pd.melt()
df.set_index 
df.reindex | df.reset_index  
dt.datetime.strptime
df_fc_comp[group_name] = pd.Categorical(df_fc_comp[group_name], categories=grp_order)
df['yoy_net_adds_smth'] = df.groupby([grp_col])['yoy_net_adds'].rolling(window=smooth_window).mean().tolist()
est_growth_rates = proxy_weights.groupby('country', as_index=False)[['weight', y_var_weighted]
df_mkd = df_comp.to_markdown(index=False, floatfmt=col_formats)
df_reduction_curve = df_fit.pivot_table(
            index=['reporting_country', 'mau_date'],
            columns='days_since',
            values=response_var,
            aggfunc='first') 
df_weights['mau_date_min'] = df_weights.groupby('reporting_country')['mau_date'].transform('min')
df_weights['day_n'] = df_weights['day_n'].dt.days + 1
#from pandas.plotting import register_matplotlib_converters
fcst_end_date = pd.to_datetime(partition) + pd.DateOffset(years=output_yrs)
pd.get_dummies()
pd.to_numeric()
df['A'].clip(lower =0, higher =25)
date_range = pd.date_range(min_date, max_date)
df.nultiply 
df_aggr['reg_mau_ratio'] = df_aggr['reg_mau_ratio'].rolling(window=smoothing_window, min_periods=1
valid_streams = valid_streams.groupby(['country'] + fields).rolling(28).mean()
"""
    df_list = []
    for index, row in df_nested.iterrows():
        temp_df = pd.DataFrame(row['forecast'])
        temp_df[grp_col] = row[grp_col]
        df_list.append(temp_df)
    df_flat = pd.concat(df_list)
    return df_flat
"""