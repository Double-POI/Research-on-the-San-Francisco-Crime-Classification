import pandas as pd
train_data = pd.read_csv('../input/sf-crime/train.csv.zip',
                         parse_dates=['Dates'])
test_data = pd.read_csv('../input/sf-crime/test.csv.zip',
                        parse_dates=['Dates'])

train_data.info()
test_data.info()

print(train_data[:5])
print(test_data[:5])

import seaborn as sns
import matplotlib.pyplot as plt

cate_group = train_data.groupby(by='Category').size()
cat_num = len(cate_group.index)
cate_group.sort_values(ascending=False, inplace=True)
cate_group.plot(kind='bar',
                logy=True,
                color=sns.color_palette('coolwarm', cat_num))
plt.savefig('picture/No. of Crime types.png', dpi=400, bbox_inches='tight')
plt.close('all')
