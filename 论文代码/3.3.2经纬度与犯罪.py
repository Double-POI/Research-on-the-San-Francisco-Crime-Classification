import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('../input/sf-crime/train.csv.zip',
                         parse_dates=['Dates'])
test_data = pd.read_csv('../input/sf-crime/test.csv.zip',
                        parse_dates=['Dates'])

xy_group = pd.concat([train_data.X, train_data.Y], axis=1)
xy_group = xy_group.drop(xy_group[xy_group.Y > 50].index)
#存在66个（-120.5,90.0）点
xy_group.plot(kind='scatter', x='X', y='Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('picture/scatter of crimes by coordinate.png',
            dpi=400,
            bbox_inches='tight')
plt.close('all')

img = plt.imread('../input/sf-crime/map.png')
dpi = 100
height, width, depth = img.shape
plt.figure(figsize=(width / dpi, height / dpi))
plt.imshow(img)
plt.axis('off')
plt.savefig('picture/map.png', dpi=dpi, bbox_inches='tight')
plt.close('all')

plt.figure(figsize=(width / dpi, height / dpi))
plt.hist2d(xy_group.X.values, xy_group.Y.values, bins=40, cmap='Reds')
plt.savefig('picture/map2.png', dpi=dpi, bbox_inches='tight')
plt.close('all')