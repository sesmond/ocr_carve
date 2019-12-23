# 参考：
# https://www.jianshu.com/p/0015058f115f
# https://www.twblogs.net/a/5b802b232b71772165a62f1d/zh-cn

import os
import cv2
import math
import  image_show
from  tuning import RotateProcessor
import numpy as np
import tifffile
from libtiff import TIFF



def distance(x1,y1,x2,y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

def lsd(img):
    # img = cv2.GaussianBlur(img, (7,7), 0)  # （5,5）表示的是卷积模板的大小，0表示的是沿x与y方向上的标准差

    # .   @param _refine The way found lines will be refined, see cv::LineSegmentDetectorModes
    # .   @param _scale The scale of the image that will be used to find the lines. Range (0..1].
    # .   @param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
    # .   @param _quant Bound to the quantization error on the gradient norm.
    # .   @param _ang_th Gradient angle tolerance in degrees.
    # .   @param _log_eps Detection threshold: -log10(NFA) \> log_eps. Used only when advance refinement
    # .   is chosen.
    # .   @param _density_th Minimal density of aligned region points in the enclosing rectangle.
    # .   @param _n_bins Number of bins in pseudo-ordering of gradient modulus.
    #TODO 必须是灰度图？
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # image_show.show(img_gray,'灰度图')

    kernel = np.ones((5, 5), np.uint8)
    img_e = cv2.erode(img_gray, kernel)  # 腐蚀
    # image_show.show(img_e,'腐蚀后')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
    img_e = cv2.dilate(img_e, kernel)  # 膨胀
    image_show.show(img_e, '二次膨胀后')
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img_e)

    line_sel =0
    new_lines = validate_lines(lines[0])
    for dline in new_lines:
        x0 = dline[0]
        y0 = dline[1]
        x1 = dline[2]
        y1 = dline[3]
        print("得出的线段：",dline)
        line_sel +=1
        cv2.line(img,(x0,y0),(x1,y1),(0,255,0),2)
    print("最终总线段条数：",line_sel)
    return img

def is_vertical(x0, y0, x1, y1):
    """
    判断是否是竖线
    :param x0:
    :param y0:
    :param x1:
    :param y1:
    :return:
    """
    #水平方向距离比数值方向距离小就算竖的
    if (abs(x1-x0)) < (abs(y1-y0)):
        return True
    return False

def validate_lines(lines):
    """
    过滤检测出来的线段
    :param lines:
    :return:
    """
    new_lines = []
    for dline in lines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        #1. 过滤掉太短的线
        if distance(x0, y0, x1, y1) < 100:
            continue
        #2. 过滤掉竖线
        if is_vertical(x0, y0, x1, y1):
            continue
        new_lines.append([x0,y0,x1,y1])
    new_lines = np.asarray(new_lines)
    # 按第二列排序
    new_lines = new_lines[np.lexsort(new_lines[:, ::-2].T)]
    H_THRES = 20
    temp_y=0
    temp_line_arr = [] # 把一行的都放到一组中然后生成一个最新的
    result_lines=[]
    for line in new_lines:
        y1 = line[1]
        y2 = line[3]
        line_h = (y1 + y2) / 2
        if temp_y==0:
            #首次把第一张放进去
            temp_y = line_h
            temp_line_arr.append(line)
        else:
            if (line_h - temp_y) <= H_THRES:
                #阈值之内的则认为在同一行
                temp_line_arr.append(line)
            else:
                #已经新起一行了则要处理掉之前那一组
                result_lines.append(conbine_lines(temp_line_arr))
                #新的一行处理
                temp_line_arr=[]
                temp_y = line_h
                temp_line_arr.append(line)
    #最后一组
    result_lines.append(conbine_lines(temp_line_arr))
    return result_lines

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


def conbine_lines(lines):
    """
    多条线合并为一条线
    :param lines:
    :return:
    """
    lines = np.asarray(lines)
    print(lines[:,0])
    #第一列
    x1 = int(np.min(lines[:,0]))
    x2 = int(np.max(lines[:,2]))
    y1 = int(np.mean(lines[:,1]))
    y2 = int(np.mean(lines[:,3]))
    return [x1,y1,x2,y2]

# def ocr(img_full_path):
#     _, img_name = os.path.split(img_full_path)
#     img_name = img_full_path
#     org_img  = cv2.imread(img_name, 0)  # 直接读为灰度图像
#     img = lsd(org_img)
#     return img


# def main():
#     import os
#
#     g = os.walk("data")



if __name__ == '__main__':

    rotate = RotateProcessor()
    file_list = list_all_files("/Users/minjianxu/Documents/task/抵押凭证OCR")
    index = 0
    tiff_list=[]
    for filename in file_list:
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            img = cv2.imread(filename)
            # image_show.show(img,'原图')
            angle,rotate_img  = rotate.process(img)
            # image_show.show(rotate_img,'微调图')
            print("微调角度：",angle)
            new_img = lsd(rotate_img)
            # image_show.show(new_img,'划线图')
            cv2.imwrite('../images/output/' + str(index) + '.jpg',new_img)
            # TIFF.write_image(out_tiff_path, new_img)

