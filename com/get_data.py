import datetime

from utils.parse_traj import ParseRawTraj, ParseMMTraj

# 获取地图上显示的n条轨迹
def get_traj_table(n=1, traj_dir='D:/flask/TrajectoryCleaning/static/download/test1/'):

    raw_dir = traj_dir + 'raw.txt'
    match_dir = traj_dir + 'match.txt'

    raw_trajs = ParseRawTraj.parse(raw_dir)
    match_trajs = ParseRawTraj.parse(match_dir)

    # 检查查询长度
    if n > len(raw_trajs):
        n = len(raw_trajs)

    r_trajs = raw_trajs[:n]
    m_trajs = match_trajs[:n]

    raw_t = []
    match_t = []

    raw_map = []
    match_map = []

    time_format = '%Y/%m/%d %H:%M:%S'

    # 轨迹序号，时间，经度，纬度 id time lng lat
    for i in range(len(r_trajs)):
        r = []
        r_m = []

        for p in r_trajs[i].pt_list:
            r_t = {}
            r_t['id'] = i
            r_t['time'] = p.time.strftime(time_format)
            r_t['lng'] = p.lng
            r_t['lat'] = p.lat
            r.append(r_t)
            r_m.append([p.lng, p.lat])

        raw_t.append(r)
        raw_map.append(r_m)

        for i in range(len(m_trajs)):
            r = []
            r_m = []

            for p in m_trajs[i].pt_list:
                r_t = {}
                r_t['id'] = i
                r_t['time'] = p.time.strftime(time_format)
                r_t['lng'] = p.lng
                r_t['lat'] = p.lat
                r.append(r_t)
                r_m.append([p.lng, p.lat])

            match_t.append(r)
            match_map.append(r_m)

    return raw_t, match_t, raw_map, match_map


# 获取轨迹统计数据
def get_traj_chart(traj_file):

    trajs = ParseRawTraj.parse(traj_file)

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
            p_num_dict[i] = p_num_dict[i] + 1

    p_num_dict = dict(sorted(p_num_dict.items(), key=lambda x: x[0]))

    length_dict = {'<1000': 0, '1000~1500': 0, '1500~2000': 0, '2000~2500': 0, '2500~3000': 0, '>3000': 0}

    for i in trajs_length:
        if i < 1000:
            length_dict['<1000'] = length_dict['<1000'] + 1
        if i >= 1000 and i < 1500:
            length_dict['1000~1500'] = length_dict['1000~1500'] + 1
        if i >= 1500 and i < 2000:
            length_dict['1500~2000'] = length_dict['1500~2000'] + 1
        if i >= 2000 and i < 2500:
            length_dict['2000~2500'] = length_dict['2000~2500'] + 1
        if i >= 2500 and i < 3000:
            length_dict['2500~3000'] = length_dict['2500~3000'] + 1
        if i >= 3000:
            length_dict['>3000'] = length_dict['>3000'] + 1

    chart_dict = {}

    # # 柱状图的label
    # print(length_dict.keys())
    # # 柱状图的数据
    # print(length_dict.values())
    #
    # # 柱状图的label
    # print(p_num_dict.keys())
    # # 柱状图的数据
    # print(p_num_dict.values())

    chart_dict['num_label'] = list(p_num_dict.keys())
    chart_dict['num_val'] = list(p_num_dict.values())
    chart_dict['length_label'] = list(length_dict.keys())
    chart_dict['length_val'] = list(length_dict.values())

    return chart_dict

