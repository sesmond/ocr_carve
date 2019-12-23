#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :TODO
@File    :   image_show.py    
@Author  : minjianxu
@Time    : 2019/12/20 3:49 下午
@Version : 1.0 
'''

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def show(img, title='无标题'):
    """
    本地测试时展示图片
    :param img:
    :param name:
    :return:
    """
    return
    font = FontProperties(fname='/Users/minjianxu/tesstutorial/font/simhei.ttf')
    plt.title(title, fontsize='large', fontweight='bold', FontProperties=font)
    plt.imshow(img)
    plt.show()

def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

if __name__ == '__main__':
    list_f =list_all_files("/Users/minjianxu/Documents/task/抵押凭证OCR")
    print(list_f)