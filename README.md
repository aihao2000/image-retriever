# ImageRetrieval

基于CLIP的图像检索系统

通过路径添加可检索的图像，并可保存检索元数据，在下次使用时直接加载

检索结果以添加时的图像路径呈现

因此一个图像添加到图像检索系统后，图像移动将导致返回的路径无效

## 环境配置

开发时使用conda虚拟环境pytorch2.0.1，可参考并运行以下命令

```shell
conda create --name "ImageRetrieval"
conda activate ImageRetrieval
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ftfy regex tqdm prettytable
pip install git+https://github.com/openai/CLIP.git
```



使用pip直接安装的方法暂未测试

```shell
pip install git+https://github.com/AisingioroHao0/ImageRetrieval.git
```

## 用法

用法可参考根目录下的lab .ipynb





