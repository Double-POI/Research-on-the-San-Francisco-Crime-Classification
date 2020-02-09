# SF-crime-Pytorch

Fight crime with Pytorch!

## 下载数据

https://www.kaggle.com/c/sf-crime/data

数据应当存放于/input中

## 写于kaggle的Notebook

https://www.kaggle.com/doublepoi/a-nn-with-residual-v2

# 总结

此程序还存在改进空间，但是家中计算资源有限，调参有很大困难，见谅。

### 备注

old（.ipynb）版本出现过拟合现象，限于Kaggle的GPU使用时间，暂不更新Notebook。

new（.py）版本加入了两个dropout层，缓解了过拟合。

新的网络结构图如下：

![net](https://github.com/Double-POI/SF-crime-Pytorch/blob/master/new-result-picture/new-result-net.jpg?raw=true)

新的网络训练过程如下：

![train](https://github.com/Double-POI/SF-crime-Pytorch/blob/master/new-result-picture/new-result-train.jpg?raw=true)

新的提交结果如下：

![submit](https://github.com/Double-POI/SF-crime-Pytorch/blob/master/new-result-picture/new-result.jpg?raw=true)
