import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('../input/sf-crime/train.csv.zip',
                         parse_dates=['Dates'])
test_data = pd.read_csv('../input/sf-crime/test.csv.zip',
                        parse_dates=['Dates'])

train_data['date'] = pd.to_datetime(train_data['Dates'])
train_data['year'] = train_data.date.dt.year
train_data['month'] = train_data.date.dt.month
train_data['day'] = train_data.date.dt.day
train_data['hour'] = train_data.date.dt.hour
train_data['minute'] = train_data.date.dt.minute

year_group = train_data.groupby('year').size()
plt.plot(year_group, 'ks-')
#2015年不完整
plt.xlabel('year')
plt.savefig('picture/No. of crimes by year.png', dpi=400, bbox_inches='tight')
plt.close('all')

year_group = train_data.groupby('year').size()
plt.plot(year_group.index[:-1], year_group[:-1], 'ks-')
#2015年不完整
plt.xlabel('year')
plt.savefig('picture/No. of crimes by year（-2015）.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')

month_group = train_data.groupby('month').size()
plt.plot(month_group, 'ks-')
plt.xlabel('month')
plt.savefig('picture/No. of crimes by month.png', dpi=400, bbox_inches='tight')
plt.close('all')

cate_group = train_data.groupby(by='Category').size()
cate_group.sort_values(ascending=False, inplace=True)
top10 = list(cate_group.index[:10])
tmp = train_data[train_data['Category'].isin(top10)]
mon_g = tmp.groupby(['Category', 'month']).size()
mon_g = mon_g.unstack()
for i in range(10):
    mon_g.iloc[i] = mon_g.iloc[i] / mon_g.sum(axis=1)[i]
mon_g.T.plot(figsize=(12, 6), style='o-')
plt.savefig('picture/No. of crimes by month(top10).png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')