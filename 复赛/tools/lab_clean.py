"""
清洗labels， 指数个股分开， 加入时间特征，最终为：
图片名称，收盘日d，收盘日d+1，收盘日d+2，收盘日d+3，过去120天最高价，过去120天最低价，
过去120天最高成交量，过去120天最低成交量，过去120天最高KDJ，过去120天最低KDJ
"""
import time
import re
import os
from datetime import datetime


def get_file_name(folder):
    output = []
    for i in os.listdir(folder):
        output.append(i.split('.')[0])
    return output


def get_week_day(time_str):
    return datetime.strftime(datetime.strptime(time_str, '%Y/%m/%d'), '%w')


def main():
    labs_file = 'revised_stock_labels（标签，如果训练集中的图片不在标签列表中，请不要使用）.csv'
    pic_dir = 'pic'
    index_pic_dir = 'index_pic'
    id_in_pic = set(get_file_name(pic_dir))
    id_in_index_pic = set(get_file_name(index_pic_dir))
    # process lab file
    with open(labs_file, 'r', encoding='utf-8') as f:
        with open('labs.txt', 'w', encoding='utf-8') as out:
            with open('index_labs.txt', 'w', encoding='utf-8') as index_out:
                f.readline()
                head = '\t'.join(['ID', 'value1', 'value2', 'value3', 'd', 'd+1', 'd+2', 'd+3', '120hP', '120lP', '120hV', '120lV', '120hKDJ', '120lKDJ'])
                out.write(head + '\n')
                index_out.write(head + '\n')
                for line in f.readlines():
                    items = line.split('\t')
                    tmp = [items[0]]  # id
                    if float(items[8]) == 0:
                        print('{} d0_open 0'.format(items[0]))
                        continue
                    tmp.append(float(items[9]) / float(items[8]) - 1.0)
                    tmp.append(float(items[10]) / float(items[9]) - 1.0)
                    tmp.append(float(items[11]) / float(items[10]) - 1.0)
                    tmp.append(get_week_day(items[4]))
                    tmp.append(get_week_day(items[5]))
                    tmp.append(get_week_day(items[6]))
                    tmp.append(get_week_day(items[7]))
                    tmp += items[24:]
                    if tmp[0] in id_in_pic:
                        out.write('\t'.join([str(i) for i in tmp]))
                    elif tmp[0] in id_in_index_pic:
                        index_out.write('\t'.join([str(i) for i in tmp]))
                    else:
                        print('{} no pic'.format(items[0]))


if __name__ == "__main__":
    main()
