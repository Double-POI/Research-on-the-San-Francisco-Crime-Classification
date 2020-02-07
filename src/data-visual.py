import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('../train.csv')

cate_group = train_data.groupby(by='Category').size()
cat_num = len(cate_group.index)
cate_group.sort_values(ascending=False,inplace=True)
cate_group.plot(kind='bar',logy=True,color=sns.color_palette('coolwarm',cat_num))
plt.title('No. of Crime types',fontsize=20)
plt.savefig('../output/picture/No. of Crime types.png',dpi=400,bbox_inches='tight')
plt.close('all')

train_data['date'] = pd.to_datetime(train_data['Dates'])
train_data['year'] = train_data.date.dt.year
train_data['month'] = train_data.date.dt.month
train_data['day'] = train_data.date.dt.day
train_data['hour'] = train_data.date.dt.hour
train_data['minute'] = train_data.date.dt.minute

year_group = train_data.groupby('year').size()
plt.plot(year_group,'ks-')
#2015年不完整
plt.xlabel('year')
plt.title('No. of crimes by year',fontsize=20)
plt.savefig('../output/picture/No. of crimes by year.png',dpi=400,bbox_inches='tight')
plt.close('all')

year_group = train_data.groupby('year').size()
plt.plot(year_group.index[:-1],year_group[:-1],'ks-')
#2015年不完整
plt.xlabel('year')
plt.title('No. of crimes by year',fontsize=20)
plt.savefig('../output/picture/No. of crimes by year（-2015）.png',dpi=400,bbox_inches='tight')
plt.close('all')

month_group = train_data.groupby('month').size()
plt.plot(month_group,'ks-')
plt.xlabel('month')
plt.title('No. of crimes by month',fontsize=20)
plt.savefig('../output/picture/No. of crimes by month.png',dpi=400,bbox_inches='tight')
plt.close('all')

top10 = list(cate_group.index[:10])
tmp = train_data[train_data['Category'].isin(top10)]
mon_g = tmp.groupby(['Category','month']).size()
mon_g = mon_g.unstack()
for i in range(10):
    mon_g.iloc[i] = mon_g.iloc[i]/mon_g.sum(axis=1)[i]
mon_g.T.plot(figsize=(12,6),style='o-')
plt.savefig('../output/picture/No. of crimes by month(top10).png',dpi=400,bbox_inches='tight')
plt.close('all')

day_group = train_data.groupby('day').size()
plt.plot(day_group,'ks-')
plt.xlabel('day')
plt.title('No. of crimes by day',fontsize=20)
plt.savefig('../output/picture/No. of crimes by day.png',dpi=400,bbox_inches='tight')
plt.close('all')

hour_group = train_data.groupby('hour').size()
plt.plot(hour_group,'ks-')
plt.xlabel('hour')
plt.title('No. of crimes by hour',fontsize=20)
plt.savefig('../output/picture/No. of crimes by hour.png',dpi=400,bbox_inches='tight')
plt.close('all')

minute_group = train_data.groupby('minute').size()
plt.plot(minute_group,'ks-')
plt.xlabel('minute')
plt.title('No. of crimes by minute',fontsize=20)
plt.savefig('../output/picture/No. of crimes by minute.png',dpi=400,bbox_inches='tight')
plt.close('all')

top6 = list(cate_group.index[:6])
tmp = train_data[train_data['Category'].isin(top6)]
hou_g = tmp.groupby(['Category','hour']).size()
hou_g = hou_g.unstack()
hou_g.T.plot(figsize=(12,6),style='o-')
plt.savefig('../output/picture/No. of crimes by hour(top6).png',dpi=400,bbox_inches='tight')
plt.close('all')

wkm = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,
    'Saturday':5,'Sunday':6}
train_data['DayOfWeek'] = train_data['DayOfWeek'].apply(lambda x: wkm[x])
tmp = train_data[train_data['Category'].isin(top6)]
wee_group = tmp.groupby(['Category','DayOfWeek']).size()
wee_group = wee_group.unstack()
wee_group.T.plot(figsize=(12,6),style='o-')
plt.xticks([0,1,2,3,4,5,6],['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])
plt.savefig('../output/picture/No. of crimes by DayOfWeek.png',dpi=400,bbox_inches='tight')
plt.close('all')

dis_group = train_data.groupby(by='PdDistrict').size()
dis_num = len(dis_group.index)
dis_group.sort_values(ascending=False,inplace=True)
dis_group.plot(kind='bar',fontsize=10,color=sns.color_palette('coolwarm',dis_num))
plt.title('No. of crimes by district',fontsize=20)
plt.savefig('../output/picture/No. of crimes by district.png',dpi=400,bbox_inches='tight')
plt.close('all')

train_data['block'] = train_data['Address'].apply(lambda x: 1 if 'block' in x.lower() else 0)
tmp = train_data[train_data['Category'].isin(top10)]
blo_group = tmp.groupby(['Category','block']).size()
blo_group.unstack().T.plot(kind='bar',figsize=(12,6),rot=45)
plt.xticks([0,1],['no block','block'])
plt.savefig('../output/picture/No. of crimes by block.png',dpi=400,bbox_inches='tight')
plt.close('all')

xy_group = pd.concat([train_data.X,train_data.Y], axis=1)
xy_group = xy_group.drop(xy_group[xy_group.Y > 50].index)
#存在66个（-120.5,90.0）点
xy_group.plot(kind='scatter',x='X',y='Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('../output/picture/scatter of crimes by coordinate.png',dpi=400,bbox_inches='tight')
plt.close('all')

img = plt.imread('../map.png')
dpi=100
height, width, depth = img.shape
plt.figure(figsize=(width / dpi, height / dpi))
plt.imshow(img)
plt.axis('off')
#plt.show()
plt.savefig('../output/picture/map.png',dpi=dpi,bbox_inches='tight')
plt.close('all')

plt.figure(figsize=(width / dpi, height / dpi))
plt.hist2d(xy_group.X.values,xy_group.Y.values,bins=40,cmap='Reds')
#plt.show()
plt.savefig('../output/picture/map2.png',dpi=dpi,bbox_inches='tight')
plt.close('all')