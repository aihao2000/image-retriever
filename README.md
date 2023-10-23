# ImageRetrieval

基于CLIP的图像检索系统

通过路径添加可检索的图像，并可保存检索元数据，在下次使用时直接加载。

可通过文本，图像进行图像检索。

支持旋转裁剪的子图搜索。

检索结果以添加时的图像路径呈现。

因此一个图像添加到图像检索系统后，图像移动将导致返回的路径无效。

可以下载lab.ipynb快速预览示例用法。

可以从百度云盘下载包含约三千五百张高分辨率测试图像的完整项目，并包含并包含推理后的元数据文件可在低计算机力的设备上直接进行检索，而非再次添加。

链接：https://pan.baidu.com/s/1ogPFGpdlztDQ7GdGdK6pZw?pwd=xcqj 
提取码：xcqj

任何问题请直接联系aihao@buaa.edu.cn，或提交issue。

## 环境配置

开发时使用conda虚拟环境pytorch2.0.1，可参考并运行以下命令

使用pip直接安装

```shell
pip install git+https://github.com/aihao2000/image-retriever
```

## 用法

用法可参考根目录下的lab .ipynb





