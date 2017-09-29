import click
import json
import numpy as np


def expect_margin(predictions, answer):
    predict_sign = np.sign(predictions)
    answer_sign = np.sign(answer)
    margin_array = []
    for m in range(answer.shape[0]):
        row = []
        for n in range(answer.shape[1]):
            a = answer[m, n]
            p = predictions[m, n]
            p_s = predict_sign[m, n]
            a_s = answer_sign[m, n]
            if p_s == a_s:
                row.append(min(abs(a), abs(p)))
            else:
                row.append(-1 * (abs(a) + abs(p)))
        margin_array.append(row)
    margin_array = np.array(margin_array)
    return np.sum(margin_array, 0)


def price_trend_hit(predictions, answer):
    predict_sign = np.sign(predictions)
    answer_sign = np.sign(answer)
    hit = np.equal(answer_sign, predict_sign)
    return np.mean(hit, 0)


def BOTscoreCal(pred, ans):
    margin1 = expect_margin(np.array(pred), np.array(ans))  # 选手预测的三日预期收益
    margin2 = expect_margin(np.array(ans), np.array(ans))  # 完美预测的三日预期收益
    price_hit = price_trend_hit(np.array(pred), np.array(ans))  # 选手预测的三日预期收益
    margin_rate = np.divide(margin1, margin2)
    BOTscore = np.sum(margin_rate)
    return BOTscore, margin_rate, price_hit


def compare(pred, ans, name='default'):
    with open(pred, encoding='utf-8') as f:
        raw_p = json.load(f)
    a = {}
    with open(ans, encoding='utf-8') as f:
        f.readline()
        for line in f.readlines():
            items = line.split('\t')
            if not items[0].endswith('jpg'):
                items[0] += '.jpg'
            a[items[0]] = [float(items[1]), float(items[2]), float(items[3])]
    p = {}
    for i in raw_p:
        if not i['uuid'].endswith('jpg'):
            i['uuid'] += '.jpg'
        p[i['uuid']] = [float(i['value1']), float(i['value2']), float(i['value3'])]
    if p.keys() != a.keys():
        print(p.keys(), a.keys())
        raise ValueError('1')
    a_list = []
    p_list = []
    for i in p:
        a_list.append(a[i])
        p_list.append(p[i])
    s = BOTscoreCal(p_list, a_list)
    print('{}: {}'.format(name, s))
    return a_list, p_list


@click.command()
@click.option('--idx_pred')
@click.option('--idx_ans')
@click.option('--pred')
@click.option('--ans')
def main(pred, ans, idx_pred, idx_ans):
    a1, p1 = compare(idx_pred, idx_ans, 'index')
    a2, p2 = compare(pred, ans, 'stocks')
    a = a1 + a2
    p = p1 + p2
    print('all: {}'.format(BOTscoreCal(p, a)))


if __name__ == '__main__':
    main()
