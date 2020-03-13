import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('../input/sf-crime/train.csv.zip',
                         parse_dates=['Dates'])
test_data = pd.read_csv('../input/sf-crime/test.csv.zip',
                        parse_dates=['Dates'])

dis_group = train_data.groupby(by='PdDistrict').size()
dis_num = len(dis_group.index)
dis_group.sort_values(ascending=False, inplace=True)
dis_group.plot(kind='bar',
               fontsize=10,
               color=sns.color_palette('coolwarm', dis_num))
plt.savefig('picture/No. of crimes by district.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')

train_data['block'] = train_data['Address'].apply(
    lambda x: 1 if 'block' in x.lower() else 0)
cate_group = train_data.groupby(by='Category').size()
cate_group.sort_values(ascending=False, inplace=True)
top10 = list(cate_group.index[:10])
tmp = train_data[train_data['Category'].isin(top10)]
blo_group = tmp.groupby(['Category', 'block']).size()
blo_group.unstack().T.plot(kind='bar', figsize=(12, 6), rot=45)
plt.xticks([0, 1], ['no block', 'block'])
plt.savefig('picture/No. of crimes by block.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')
