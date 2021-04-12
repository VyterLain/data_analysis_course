import pandas as pd

df = pd.read_csv('./ex210412/2.csv', encoding='GB18030')
g = df.groupby('订单号')

f = open('./ex210412/output.txt', 'w+')
s = ''
for key, gk in g:
    s = s + str(key) + ' '
    for item in gk['货品编号']:
        s = s + str(item) + ','
    s = s[:-1]
    s = s + '\n'

f.write(s)
f.close()
