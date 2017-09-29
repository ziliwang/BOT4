import numpy as np


def parse(d, pl=0.1, ph=450.0):
    """解析预处理"""
    x = []
    low = pl
    high = ph
    for i, x1 in enumerate(d['x']):
        tmp = np.zeros(7)
        tmp[int(d['d+1'][i])] = 1.0
        x2 = tmp
        tmp = np.zeros(7)
        tmp[int(d['d+2'][i])] = 1.0
        x2 = np.hstack((x2, tmp))
        tmp = np.zeros(7)
        tmp[int(d['d+3'][i])] = 1.0
        x2 = np.hstack((x2, tmp))
        lp = (float(d['120lP'][i]) - low) / (high - low)
        hp = (float(d['120hP'][i]) - low) / (high - low)
        x2 = np.hstack((x2, np.array([lp, hp])))
        x.append([x1, x2])
    y = np.vstack((np.array(d['value1']),
                   np.array(d['value2']),
                   np.array(d['value3'])))
    y = y.astype(np.float32)
    y = y.T
    return x, y


def test_parse(d, pl=0.1, ph=450.0):
    """解析预处理"""
    x = []
    _id = []
    low = pl
    high = ph
    for i, x1 in enumerate(d['x']):
        tmp = np.zeros(7)
        tmp[int(d['d+1'][i])] = 1.0
        x2 = tmp
        tmp = np.zeros(7)
        tmp[int(d['d+2'][i])] = 1.0
        x2 = np.hstack((x2, tmp))
        tmp = np.zeros(7)
        tmp[int(d['d+3'][i])] = 1.0
        x2 = np.hstack((x2, tmp))
        lp = (float(d['120lP'][i]) - low) / (high - low)
        hp = (float(d['120hP'][i]) - low) / (high - low)
        x2 = np.hstack((x2, np.array([lp, hp])))
        x.append([x1, x2])
        _id.append(d['id'][i])
    return x, _id
