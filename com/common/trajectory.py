from .spatial_func import distance, SPoint, cal_loc_along_line
from .mbr import MBR
from datetime import timedelta


class STPoint(SPoint):

    def __init__(self, lat, lng, time, data=None):
        super(STPoint, self).__init__(lat, lng)
        self.time = time
        self.data = data  # contains edge's attributes

    def __str__(self):
        """
        For easily reading the output
        """
        # __repr__() to change the print review
        # st = STPoint()
        # print(st) will not be the reference but the following format
        # if __repr__ is changed to str format, __str__ will be automatically change.

        return str(self.__dict__)  # key and value of self attributes
        # return '({}, {}, {})'.format(self.time.strftime('%Y/%m/%d %H:%M:%S'), self.lat, self.lng)


class Trajectory:

    def __init__(self, oid, tid, pt_list):
        self.oid = oid
        self.tid = tid
        self.pt_list = pt_list

    # 轨迹的总时间 s
    def get_duration(self):

        return (self.pt_list[-1].time - self.pt_list[0].time).total_seconds()

    # 轨迹总长度 m
    def get_distance(self):
        dist = 0.0
        pre_pt = self.pt_list[0]
        for pt in self.pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist += tmp_dist
            pre_pt = pt
        return dist

    # gps轨迹点之间的平均间隔时间 s
    def get_avg_time_interval(self):
        point_time_interval = []
        # How clever method! zip to get time interval

        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_time_interval.append((cur.time - pre.time).total_seconds())
        return sum(point_time_interval) / len(point_time_interval)

    # GPS轨迹点的平均距离 m
    def get_avg_distance_interval(self):
        point_dist_interval = []
        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_dist_interval.append(distance(pre, cur))
        return sum(point_dist_interval) / len(point_dist_interval)

    # 轨迹序列生成的边界
    def get_mbr(self):
        return MBR.cal_mbr(self.pt_list)

    # 轨迹起始时间
    def get_start_time(self):
        return self.pt_list[0].time

    # 轨迹点结束时间
    def get_end_time(self):
        return self.pt_list[-1].time

    # 轨迹中点时间
    def get_mid_time(self):
        return self.pt_list[0].time + (self.pt_list[-1].time - self.pt_list[0].time) / 2.0

    # 轨迹的中点(空间几何上的中点坐标)
    def get_centroid(self):
        mean_lat = 0.0
        mean_lng = 0.0
        for pt in self.pt_list:
            mean_lat += pt.lat
            mean_lng += pt.lng
        mean_lat /= len(self.pt_list)
        mean_lng /= len(self.pt_list)
        return SPoint(mean_lat, mean_lng)

    # 得到时间范围内的轨迹子序列
    def query_trajectory_by_temporal_range(self, start_time, end_time):
        # start_time <= pt.time < end_time
        traj_start_time = self.get_start_time()
        traj_end_time = self.get_end_time()
        if start_time > traj_end_time:
            return None
        if end_time <= traj_start_time:
            return None
        st = max(traj_start_time, start_time)
        et = min(traj_end_time + timedelta(seconds=1), end_time)
        start_idx = self.binary_search_idx(st)  # pt_list[start_idx].time <= st < pt_list[start_idx+1].time
        if self.pt_list[start_idx].time < st:
            # then the start_idx is out of the range, we need to increase it
            start_idx += 1
        end_idx = self.binary_search_idx(et)  # pt_list[end_idx].time <= et < pt_list[end_idx+1].time
        if self.pt_list[end_idx].time < et:
            # then the end_idx is acceptable
            end_idx += 1
        sub_pt_list = self.pt_list[start_idx:end_idx]
        return Trajectory(self.oid, get_tid(self.oid, sub_pt_list), sub_pt_list)

    # 查询指定时间在轨迹的第几个点
    def binary_search_idx(self, time):
        # self.pt_list[idx].time <= time < self.pt_list[idx+1].time
        # if time < self.pt_list[0].time, return -1
        # if time >= self.pt_list[len(self.pt_list)-1].time, return len(self.pt_list)-1
        nb_pts = len(self.pt_list)
        if time < self.pt_list[0].time:
            return -1
        if time >= self.pt_list[-1].time:
            return nb_pts - 1
        # the time is in the middle
        left_idx = 0
        right_idx = nb_pts - 1
        while left_idx <= right_idx:
            mid_idx = int((left_idx + right_idx) / 2)
            if mid_idx < nb_pts - 1 and self.pt_list[mid_idx].time <= time < self.pt_list[mid_idx + 1].time:
                return mid_idx
            elif self.pt_list[mid_idx].time < time:
                left_idx = mid_idx + 1
            else:
                right_idx = mid_idx - 1

    # 跟据时间得到轨迹上的坐标
    def query_location_by_timestamp(self, time):
        idx = self.binary_search_idx(time)
        if idx == -1 or idx == len(self.pt_list) - 1:
            return None
        if self.pt_list[idx].time == time or (self.pt_list[idx + 1].time - self.pt_list[idx].time).total_seconds() == 0:
            return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
        else:
            # interpolate location
            dist_ab = distance(self.pt_list[idx], self.pt_list[idx + 1])
            if dist_ab == 0:
                return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
            dist_traveled = dist_ab * (time - self.pt_list[idx].time).total_seconds() / \
                            (self.pt_list[idx + 1].time - self.pt_list[idx].time).total_seconds()
            return cal_loc_along_line(self.pt_list[idx], self.pt_list[idx + 1], dist_traveled / dist_ab)

    # wkt格式表示
    def to_wkt(self):
        wkt = 'LINESTRING ('
        for pt in self.pt_list:
            wkt += '{} {}, '.format(pt.lng, pt.lat)
        wkt = wkt[:-2] + ')'
        return wkt

    def __hash__(self):
        return hash(self.oid + '_' + self.pt_list[0].time.strftime('%Y%m%d%H%M%S') + '_' +
                    self.pt_list[-1].time.strftime('%Y%m%d%H%M%S'))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return f'Trajectory(oid={self.oid},tid={self.tid})'

# 生成轨迹id (tid)
def get_tid(oid, pt_list):
    return oid + '_' + pt_list[0].time.strftime('%Y%m%d%H%M%S') + '_' + pt_list[-1].time.strftime('%Y%m%d%H%M%S')

