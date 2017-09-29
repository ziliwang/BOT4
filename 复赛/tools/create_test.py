import os


index_file = "raw/复赛第二次提交测试集-标签（指数）.txt"
_file = "raw/复赛第二次提交测试集-标签（个股）.txt"
pic_dir = 'pic'
index_pic_dir = 'index_pic'


def get_file_name(folder):
    output = []
    for i in os.listdir(folder):
        output.append(i)
    return output


id_in_index_pic = set(get_file_name(index_pic_dir))
with open(index_file, encoding='utf-8') as f:
    d = [i.split('\t') for i in f.read().split('\n') if i]
with open('test_index_labs.txt', 'w', encoding='utf-8') as f:
    head = '\t'.join(['ID', 'd', 'd+1', 'd+2', 'd+3', '120hP', '120lP', '120hV', '120lV', '120hKDJ', '120lKDJ'])
    f.write(head + '\n')
    for i in d[1:]:
        if i[0] in id_in_index_pic:
            f.write('\t'.join(i) + '\n')
        else:
            print('{} not exists'.format(d[0]))

id_in_pic = set(get_file_name(pic_dir))
with open(_file, encoding='utf-8') as f:
    d = [i.split('\t') for i in f.read().split('\n') if i]
with open('test_labs.txt', 'w', encoding='utf-8') as f:
    head = '\t'.join(['ID', 'd', 'd+1', 'd+2', 'd+3', '120hP', '120lP', '120hV', '120lV', '120hKDJ', '120lKDJ'])
    f.write(head + '\n')
    for i in d[1:]:
        if i[0] in id_in_pic:
            f.write('\t'.join(i) + '\n')
        else:
            print('{} not exists'.format(d[0]))
