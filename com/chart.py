from utils.parse_traj import ParseMMTraj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # 训练结果的数据分析
#
# df = pd.read_json('data/loss/train_loss.json')
#
# # 获取列名
# column = df.columns.values
#
#
# index_loss = [i*5 for i in range(int(len(list(df['train_ttl_loss']))/5))]
# recall_loss = [list(df['train_id_recall'])[i] for i in index_loss]
# precision_loss = [list(df['train_id_precision'])[i] for i in index_loss]
# rid_loss = [list(df['train_id_loss'])[i] for i in index_loss]
# rate_loss = [list(df['train_rate_loss'])[i] for i in index_loss]
# ttl_loss = [list(df['train_ttl_loss'])[i] for i in index_loss]
#
# print(column)
# print(index_loss)
# print(recall_loss)
# print(precision_loss)
# print(rid_loss)
# print(rate_loss)
# print(ttl_loss)

# for c in column:
#     print(list(df[c]))




trajs = ParseMMTraj.parse('data/model/model_data/test_data/test_raw_trajectory_1.txt')

# 轨迹数目
traj_num = len(trajs)

# 每条轨迹的轨迹点数目
p_num = []

# 每条轨迹轨迹长度
trajs_length = []

# 平均距离
p_avg_dist = []


for t in trajs:
    # 每条轨迹轨迹点数目
    p_num.append(len(t.pt_list))

    # 轨迹长度
    trajs_length.append(t.get_distance())

    # 平均距离
    p_avg_dist.append(t.get_avg_distance_interval())

# 字典统计
p_num_dict = {}

for i in p_num:
    if i not in p_num_dict.keys():
        p_num_dict[i] = 0
    else:
        p_num_dict[i] = p_num_dict[i]+1


p_num_dict = dict(sorted(p_num_dict.items(), key=lambda x: x[0]))
# 柱状图的label
print(p_num_dict.keys())
# 柱状图的数据
print(p_num_dict.values())

# print(p_num)
# print(trajs_duration)
# print(p_avg_dist)

# 4062.6651930017542 33.58652471703318
# <1000 1000~1500 1500~2000 2000~2500 2500~3000 3000~3500 3500~4000 >4000
# print(max(trajs_length), min(trajs_length))

length_dict = {'<1000':0, '1000~1500':0, '1500~2000':0, '2000~2500':0, '2500~3000':0, '>3000':0}

for i in trajs_length:
    if i<1000:
        length_dict['<1000'] = length_dict['<1000']+1
    if i>=1000 and i<1500:
        length_dict['1000~1500'] = length_dict['1000~1500'] + 1
    if i>=1500 and i<2000:
        length_dict['1500~2000'] = length_dict['1500~2000'] + 1
    if i>=2000 and i<2500:
        length_dict['2000~2500'] = length_dict['2000~2500'] + 1
    if i>=2500 and i<3000:
        length_dict['2500~3000'] = length_dict['2500~3000'] + 1
    if i>=3000:
        length_dict['>3000'] = length_dict['>3000'] + 1

# 柱状图的label
print(length_dict.keys())
# 柱状图的数据
print(length_dict.values())

