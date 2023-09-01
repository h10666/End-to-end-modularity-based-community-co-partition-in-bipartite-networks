import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

col = ['Database', 'ACC', 'beta']
data = [['ACM', 0.8909, 0.1],
        ['ACM', 0.9308, 0.5],
        ['ACM', 0.8164, 1],
        ['DBLP', 0.7784, 0.1],
        ['DBLP', 0.7987, 0.5],
        ['DBLP', 0.6759, 1],
        ['Amazon-Photo', 0.7201, 0.1],
        ['Amazon-Photo', 0.7385, 0.5],
        ['Amazon-Photo', 0.6471, 1],
        ['Amazon-Computer', 0.5819, 0.1],
        ['Amazon-Computer', 0.618, 0.5],
        ['Amazon-Computer', 0.5105, 1],
        ]
df = pd.DataFrame(data=data, columns=col)
plt.figure(dpi=300, figsize=(8,6))
sns.barplot(x=df['Database'], y=df['ACC'], hue=df['beta'],)
plt.show()


# col_hard = ['ACC', 'method', 'epoch', 'line']
# data_hrad = [[0.8961, 'GCA', 500, 'origin'],
# [0.8132, 'GCA', 400, 'origin'],
# [0.7161, 'GCA', 300, 'origin'],
# [0.6328, 'GCA', 200, 'origin'],
# [0.4273, 'GCA', 100, 'origin'],
# [0, 'GCA', 0, 'origin'],
# [0.9047, 'GCA', 500, 'origin+Hard'],
# [0.8415, 'GCA', 400, 'origin+Hard'],
# [0.7430, 'GCA', 300, 'origin+Hard'],
# [0.6602, 'GCA', 200, 'origin+Hard'],
# [0.3878, 'GCA', 100, 'origin+Hard'],
# [0, 'GCA', 0, 'origin+Hard'],
#
# [0.9016, 'DGI', 500, 'origin'],
# [0.8405, 'DGI', 400, 'origin'],
# [0.7519, 'DGI', 300, 'origin'],
# [0.6458, 'DGI', 200, 'origin'],
# [0.3273, 'DGI', 100, 'origin'],
# [0, 'DGI', 0, 'origin'],
# [0.9159, 'DGI', 500, 'origin+Hard'],
# [0.8547, 'DGI', 400, 'origin+Hard'],
# [0.777, 'DGI', 300, 'origin+Hard'],
# [0.6674, 'DGI', 200, 'origin+Hard'],
# [0.3356, 'DGI', 100, 'origin+Hard'],
# [0, 'DGI', 0, 'origin+Hard'],
# [0.9142, 'GraphCL', 500, 'origin'],
# [0.8844, 'GraphCL', 400, 'origin'],
# [0.8021, 'GraphCL', 300, 'origin'],
# [0.6556, 'GraphCL', 200, 'origin'],
# [0.4152, 'GraphCL', 100, 'origin'],
# [0, 'GraphCL', 0, 'origin'],
# [0.9174, 'GraphCL', 500, 'origin+Hard'],
# [0.8924, 'GraphCL', 400, 'origin+Hard'],
# [0.8127, 'GraphCL', 300, 'origin+Hard'],
# [0.678, 'GraphCL', 200, 'origin+Hard'],
# [0.4663, 'GraphCL', 100, 'origin+Hard'],
# [0, 'GraphCL', 0, 'origin+Hard'],
#              ]
# df_hard = pd.DataFrame(data=data_hrad, columns=col_hard)
# plt.figure(dpi=300, figsize=(8,6))
# sns.lineplot(data=df_hard, x='epoch', y='ACC', hue='method', style='line')
# plt.show()



