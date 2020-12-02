# Im2Latex 图片转换成Latex公式

![License](https://img.shields.io/apm/l/vim-mode.svg)

Deep CNN编码器+ LSTM注意力解码器，图片转换成Latex公式，Image to Latex，该模型是pytorch实现的  [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html)


## 运行的样本结果展示



![sample_result](imgs/sample_result.png)





## 在IM2LATEX-100K测试数据集上的实验结果 

| BLUE-4 | Edit Distance | Exact Match |
| ------ | ------------- | ----------- |
| 40.80  | 44.23         | 0.27        |



## 开始



**安装依赖:**
对依赖要求不严格，任意版本均可
```bash
pip install -r requirement.txt
```

**下载数据集进行训练:**

```bash
cd data
wget http://lstm.seas.harvard.edu/latex/data/im2latex_validate_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_train_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_test_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/formula_images_processed.tar.gz
wget http://lstm.seas.harvard.edu/latex/data/im2latex_formulas.norm.lst
tar -zxvf formula_images_processed.tar.gz
```

**预处理:**

读取测试集，验证集，训练集，保存成图片tensor和公式文本对的形式，做成一个list，保存成pkl

```bash
python preprocess.py

生成：train文件有13GB
-rw-r--r--      1 admin  staff   2.6G Dec  2 13:07 test.pkl
-rw-r--r--      1 admin  staff    13G Dec  2 13:10 train.pkl
-rw-r--r--      1 admin  staff   1.4G Dec  2 13:06 validate.pkl
-rw-r--r--      1 admin  staff   6.2K Dec  2 13:12 vocab.pkl

```


**构建 vocab**
```bash
# 在data目录下生成 vocab.pkl
python build_vocab.py
```

**训练:**

```
python train.py --data_path=data --save_dir=output --dropout=0.2 --add_position_features --epoches=25 --max_len=150
```
     python train.py \
          --data_path=[data dir] \
          --save_dir=[the dir for saving ckpts] \
          --dropout=0.2 --add_position_features \
          --epoches=25 --max_len=150
**评估:**

```buildoutcfg
python evaluate.py --split=test --model_path=output --data_path=data --batch_size=32
```

```bash
python evaluate.py --split=test \
     --model_path=[the path to model] \
     --data_path=[data dir] \
     --batch_size=32 \
     --ref_path=[the file to store reference] \    #保存引用的位置
     --result_path=[the file to store decoding result]   #保存生成结果
```



## Features

- [x] Schedule Sampling from [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/pdf/1506.03099.pdf)
- [x] Positional Embedding from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [x] Batch beam search
- [x] Training from checkpoint 
- [ ] Improve the code of data loading for cpu/cuda memery efficiency 
- [ ] **Finetune hyper parameters for better performance**
- [ ] A HTML Page allowing upload picture to decode

































