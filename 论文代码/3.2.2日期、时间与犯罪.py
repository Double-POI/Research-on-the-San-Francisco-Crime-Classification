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

day_group = train_data.groupby('day').size()
plt.plot(day_group, 'ks-')
plt.xlabel('day')
plt.savefig('picture/No. of crimes by day.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')

hour_group = train_data.groupby('hour').size()
plt.plot(hour_group, 'ks-')
plt.xlabel('hour')
plt.savefig('picture/No. of crimes by hour.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')

minute_group = train_data.groupby('minute').size()
plt.plot(minute_group, 'ks-')
plt.xlabel('minute')
plt.savefig('picture/No. of crimes by minute.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')

cate_group = train_data.groupby(by='Category').size()
cate_group.sort_values(ascending=False, inplace=True)
top6 = list(cate_group.index[:6])
tmp = train_data[train_data['Category'].isin(top6)]
hou_g = tmp.groupby(['Category', 'hour']).size()
hou_g = hou_g.unstack()
hou_g.T.plot(figsize=(12, 6), style='o-')
plt.savefig('picture/No. of crimes by hour(top6).png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')

wkm = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
train_data['DayOfWeek'] = train_data['DayOfWeek'].apply(lambda x: wkm[x])
tmp = train_data[train_data['Category'].isin(top6)]
wee_group = tmp.groupby(['Category', 'DayOfWeek']).size()
wee_group = wee_group.unstack()
wee_group.T.plot(figsize=(12, 6), style='o-')
plt.xticks([0, 1, 2, 3, 4, 5, 6],
           ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
plt.savefig('picture/No. of crimes by DayOfWeek.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')