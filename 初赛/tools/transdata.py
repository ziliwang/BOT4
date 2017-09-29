import json
import pickle
from news_submission import *
import crash

files = ['pricedetail.json',
         'AnnouncementsRelations.json',
         'AnnouncementsTrainSample.json',
         'ResearchRelation.json',
         'ResearchTrainSample.json',
        'cNewsRelations.json',
        'ccNewsTrainSample.json']

data = {}


class CJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, float):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, str(obj))


def preload_data():
    """
    将初赛提供之json数据根据files的定义读入内存，并且统一透过名为data的dict统一存放，以便日后叫用
    """
    global files, data
    for _file in files[:-2]:
    # for _file in files:
        print('loading {}'.format(_file))
        with open(_file, encoding='utf-8') as f:
            data[_file] = json.loads(f.read().replace('\n', '')[1:])
    for _file in files[-2:]:
        print('loading {}'.format(_file))
        with open(_file, encoding='utf-8') as f:
            data[_file] = json.load(f)


class CJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, float):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, str(obj))


def gen_base():
    """
    产生固化的测试集数据以及提交数据格式(由训练集数据格式转换)
    :return  newssub:测试集数据
    """
    preload_data()
    for i in data:
        print('{} Keys: {}'.format(i, list(data[i][0].keys())))
    newssub = news_submission()
    newssub.load_items(data['pricedetail.json'])
    newssub.load_news(data['cNewsRelations.json'], data['ccNewsTrainSample.json'])
    newssub.load_researches(data['ResearchRelation.json'], data['ResearchTrainSample.json'])
    newssub.load_annoncements(data['AnnouncementsRelations.json'], data['AnnouncementsTrainSample.json'])

    with open('trainset.pkl', 'wb') as handle:
            pickle.dump(newssub, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    gen_base()
