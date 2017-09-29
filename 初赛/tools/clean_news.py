from bs4 import BeautifulSoup
import re
import json

with open('NewsTrainSample.json', encoding='utf-8') as f:
    d = json.loads(f.read()[1:])
for i in d:
    if 'content' in i:
        tmp = i['content']
        i['content'] = re.sub(r'\u3000', '', BeautifulSoup(tmp).text)

with open('cNewsTrainSample.json', 'w', encoding='utf-8') as f:
    json.dump(d, f)
