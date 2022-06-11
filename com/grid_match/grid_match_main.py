from common.mbr import MBR
from common.spatial_func import SPoint, project_pt_to_segment, LAT_PER_METER, LNG_PER_METER
from utils.parse_traj import ParseRawTraj
from common.trajectory import Trajectory,STPoint
from map_matching.candidate_point import CandidatePoint
from utils.save_traj import SaveTraj2MM
from utils.model_utils import load_rn_dict, get_rid_grid


# 设置网格大小为50m
grid_size = 50

# 加载路网字典
rn_dict = load_rn_dict('../data/','porto_network_dict.json')

# 路网字典对应的最大区域边界
# road_network的mbr
# m = MBR(36.5361256, 116.7658218, 37.1456445, 117.5099988)

# porto的mbr （-8.65~-8.55,41.1~41.2)（lng，lat）
m = MBR(41.1,-8.65,41.2,-8.55)

# 加载网格与路网字典映射的字典表
grid_dict, mx, my = get_rid_grid(m, grid_size, rn_dict)

# print('mx:',mx, ' my:', my)

# print(str(grid_dict))
#
#
# with open("rn_grid_dict_shenzhen.json", "w", encoding="utf-8") as f:
#
#     f.write(str(grid_dict))

# lat, lng 测试点
# point = SPoint(22.5376,114.014702)

# 每个单元格的长度宽度
lat_unit = LAT_PER_METER * grid_size
lng_unit = LNG_PER_METER * grid_size

# # 计算坐标点所在网格
# locgrid_x = max(1, int((point.lat - m.min_lat) / lat_unit) + 1)
# locgrid_y = max(1, int((point.lng - m.min_lng) / lng_unit) + 1)
#
# # 坐标点网格所在
# print('(x,y):',locgrid_x, ',', locgrid_y)
#
# # 获取网格所在的道路id
# l_rid = grid_dict[(locgrid_x,locgrid_y)]
# print('list of road id:', l_rid)
#
# # 跟据道路id,获取道路点进行匹配计算
# for l in l_rid:
#     l_o, l_d = rn_dict[l]['coords'][0], rn_dict[l]['coords'][1]
#     projection, rate, dist = project_pt_to_segment(l_o, l_d, point)
#     print('projection, rate, dist:',projection,',', rate,',', dist)

# 加载轨迹
raw_traj = ParseRawTraj.parse('../data/porto_raw_data.txt')

m_traj = []

# Trajectory oid, tid, pt_list

for t in raw_traj:
    oid = t.oid
    tid = t.tid
    pt_list = []

    for p in t.pt_list:
        # 计算每个坐标点对应的匹配坐标
        point = SPoint(p.lat, p.lng)

        # 计算坐标点所在网格
        locgrid_x = max(1, int((point.lat - m.min_lat) / lat_unit) + 1)
        locgrid_y = max(1, int((point.lng - m.min_lng) / lng_unit) + 1)

        # 坐标点网格所在
        # print('(x,y):', locgrid_x, ',', locgrid_y)

        # 查看索引是否存在,不存在就视为噪点去掉 -- 丢弃法
        # if (locgrid_x, locgrid_y) not in grid_dict.keys():
        #     continue

        l_rid = []

        # 获取大网格所有的路段id集合
        if (locgrid_x, locgrid_y) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x, locgrid_y)]
        if (locgrid_x-1, locgrid_y) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x-1, locgrid_y)]
        if (locgrid_x+1, locgrid_y) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x+1, locgrid_y)]
        if (locgrid_x, locgrid_y-1) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x, locgrid_y-1)]
        if (locgrid_x, locgrid_y+1) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x, locgrid_y+1)]
        if (locgrid_x-1, locgrid_y-1) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x-1, locgrid_y-1)]
        if (locgrid_x-1, locgrid_y+1) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x-1, locgrid_y+1)]
        if (locgrid_x+1, locgrid_y-1) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x+1, locgrid_y-1)]
        if (locgrid_x+1, locgrid_y+1) in grid_dict.keys():
            l_rid = l_rid + grid_dict[(locgrid_x+1, locgrid_y+1)]

        l_rid = list(set(l_rid))

        # 如果上下左右网格都没有路段，则剔除该点
        if len(l_rid) == 0:
            continue

        # 获取网格所在的道路id
        # l_rid = grid_dict[(locgrid_x, locgrid_y)]
        # print('list of road id:', l_rid)

        # 匹配到的点
        pro = None
        eid = 0
        error = 0.0
        offset = 10000
        r = 0.0

        # 跟据道路id,获取道路点进行匹配计算
        for l in l_rid:
            l_o, l_d = rn_dict[l]['coords'][0], rn_dict[l]['coords'][1]
            projection, rate, dist = project_pt_to_segment(l_o, l_d, point)
            # print('projection, rate, dist:', projection, ',', rate, ',', dist)
            if dist < offset:
                pro = projection
                eid = l
                error = dist
                offset = dist
                r = rate

        # 构建匹配的候选点
        if pro is not None:
            candi_pt = CandidatePoint(pro.lat, pro.lng, eid, error, offset, r)
        else:
            candi_pt = None
        pt = STPoint(point.lat, point.lng, p.time, {'candi_pt': candi_pt})
        pt_list.append(pt)

    # 添加轨迹
    traj = Trajectory(oid, tid, pt_list)
    m_traj.append(traj)

    print(oid)

save = SaveTraj2MM('WGS84ToGCJ02')
save.store(m_traj, '../data/grid_match_data/grid_match_porto_data.txt')

