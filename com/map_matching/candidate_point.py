from common.spatial_func import SPoint, LAT_PER_METER, LNG_PER_METER, project_pt_to_segment, distance

# 候选点
class CandidatePoint(SPoint):
    def __init__(self, lat, lng, eid, error, offset, rate):
        super(CandidatePoint, self).__init__(lat, lng)
        self.eid = eid
        self.error = error
        self.offset = offset
        self.rate = rate

    def __str__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)

    def __repr__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)

    def __hash__(self):
        return hash(self.__str__())
    

# 获取候选点集 传入：搜索点（中心点）、路网、搜索距离
def get_candidates(pt, rn_dict, grid_dict,mbr, search_dist):

    candidates = None

    # # 候选矩形框
    # mbr = MBR(pt.lat - search_dist * LAT_PER_METER,
    #           pt.lng - search_dist * LNG_PER_METER,
    #           pt.lat + search_dist * LAT_PER_METER,
    #           pt.lng + search_dist * LNG_PER_METER)
    # 
    # # 跟据矩形窗口搜索边 查询矩形框内的路段集合
    # candidate_edges = rn.range_query(mbr)  # list of edges (two nodes/points)

    # 每个单元格的长度宽度
    lat_unit = LAT_PER_METER * search_dist
    lng_unit = LNG_PER_METER * search_dist

    # 计算坐标点所在网格
    locgrid_x = max(1, int((pt.lat - mbr.min_lat) / lat_unit) + 1)
    locgrid_y = max(1, int((pt.lng - mbr.min_lng) / lng_unit) + 1)

    candidate_edges = []

    # 获取大网格所有的路段id集合
    if (locgrid_x, locgrid_y) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x, locgrid_y)]
    if (locgrid_x - 1, locgrid_y) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x - 1, locgrid_y)]
    if (locgrid_x + 1, locgrid_y) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x + 1, locgrid_y)]
    if (locgrid_x, locgrid_y - 1) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x, locgrid_y - 1)]
    if (locgrid_x, locgrid_y + 1) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x, locgrid_y + 1)]
    if (locgrid_x - 1, locgrid_y - 1) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x - 1, locgrid_y - 1)]
    if (locgrid_x - 1, locgrid_y + 1) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x - 1, locgrid_y + 1)]
    if (locgrid_x + 1, locgrid_y - 1) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x + 1, locgrid_y - 1)]
    if (locgrid_x + 1, locgrid_y + 1) in grid_dict.keys():
        candidate_edges = candidate_edges + grid_dict[(locgrid_x + 1, locgrid_y + 1)]

    candidate_edges = list(set(candidate_edges))

    if len(candidate_edges) > 0:
        # 计算所有符合范围要求的边的投影点
        candi_pt_list = [cal_candidate_point(pt, rn_dict, candidate_edge) for candidate_edge in candidate_edges]
        # refinement
        # 筛选搜索范围距离内的候选点
        candi_pt_list = [candi_pt for candi_pt in candi_pt_list if candi_pt.error <= search_dist]

        if len(candi_pt_list) > 0:
            candidates = candi_pt_list
    return candidates

# 计算候选点 以及其属性
def cal_candidate_point(raw_pt, rn_dict, edge):

    # u, v = edge
    # coords = rn[u][v]['coords']  # GPS points in road segment, may be larger than 2

    # 获取边的点集 列表形式 [[lat,lng],[lat,lng]]
    coords = rn_dict[edge]['coords']

    # 计算投影的候选点
    candidates = [project_pt_to_segment(coords[i], coords[i + 1], raw_pt) for i in range(len(coords) - 1)]
    idx, (projection, coor_rate, dist) = min(enumerate(candidates), key=lambda x: x[1][2])
    # enumerate return idx and (), x[1] --> () x[1][2] --> dist. get smallest error project edge
    offset = 0.0
    # 计算匹配点与道路起始点的距离
    for i in range(idx):
        offset += distance(coords[i], coords[i + 1])  # make the road distance more accurately
    offset += distance(coords[idx], projection)  # distance of road start position and projected point

    # if rn[u][v]['length'] == 0:
    #     rate = 0
    #     # print(u, v)
    # else:
    #     rate = offset/rn[u][v]['length']  # rate of whole road, coor_rate is the rate of coords.
    # return CandidatePoint(projection.lat, projection.lng, rn[u][v]['eid'], dist, offset, rate)

    # 计算移动比率
    if rn_dict[edge]['length'] == 0:
        rate = 0
        # print(u, v)
    else:
        rate = offset/rn_dict[edge]['length']  # rate of whole road, coor_rate is the rate of coords.
    # 得到候选点
    return CandidatePoint(projection.lat, projection.lng, edge, dist, offset, rate)

