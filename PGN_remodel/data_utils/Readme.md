## 此文件为data_utils说明文件
>1 config.py 为各种存储路径的配置文件<br>
>2 file_utils.py 为存储词典的工具
>>2.1 新增构造文件名的函数

>3 wv_loader.py a.根据训练好的w2v模型加载词典b.建立正反向词典c.读取正反向词典<br>
>4 mutli_proc_utils.py 对df文件进行多线程处理<br>
>5 data_loader2.py 对数据进行预处理
>>a.数据加载去空 b.数据的清洗、停用词以及切词分词处理 c.合并训练测试集用于词向量训练 d.利用训练好的词向量模型进行填充进行重新训练以获取新的词典 e.利用词向量模型去获取词向量矩阵 f.最后对文本进行索引转化<br>
12.25<br>
>6 新增gpu_utils.py文件查看是否有可用GPU<br>
>7 新增params_utils.py文件去汇总一些网络中参数配置信息便于调参<br>
>8 新增wv_loader.py文件中直接读取词向量矩阵的函数<br>
>9 