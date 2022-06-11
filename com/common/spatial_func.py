import math

# 地球平均半径(米)
EARTH_MEAN_RADIUS_METER = 6371008.7714

# 每米在经纬度上有多少度
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05

"""
# 角度和弧度转换
DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS

# 角度计算弧长的公式系数 弧长=n*pi*r/180 n为度数
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER


"""


# 坐标点类
class SPoint:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    # 用于将值转化为适于人阅读的形式
    # 简单的调用方法 : str(obj)
    def __str__(self):
        return '({},{})'.format(self.lat, self.lng)

    # 转化为供解释器读取的形式
    # 简单的调用方法 : repr(obj)
    def __repr__(self):
        return self.__str__()

    # 比较两个坐标点是否相等
    def __eq__(self, other):
        return self.lat == other.lat and self.lng == other.lng

    # 比较两个坐标点是否不相等
    def __ne__(self, other):
        return not self == other

"""
# 判断坐标是否相等 也可以用 ==
def same_coords(a, b):
    return a == b
"""



# 计算a和b的距离 用haversine公式计算 返回距离（米）
# http://www.movable-type.co.uk/scripts/latlong.html
"""
const R = 6371e3; // metres
const φ1 = lat1 * Math.PI/180; // φ, λ in radians
const φ2 = lat2 * Math.PI/180;
const Δφ = (lat2-lat1) * Math.PI/180;
const Δλ = (lon2-lon1) * Math.PI/180;

const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
          Math.cos(φ1) * Math.cos(φ2) *
          Math.sin(Δλ/2) * Math.sin(Δλ/2);
const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

const d = R * c; // in metres
"""
def distance(a, b):
    # 如果坐标相等
    if a == b:
        return 0.0

    # 纬度的弧度表示
    a_lat = math.radians(a.lat)
    b_lat = math.radians(b.lat)

    # 得到两个坐标差的弧度值
    delta_lat = math.radians(b.lat - a.lat)
    delta_lng = math.radians(b.lng - a.lng)

    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(a_lat) * math.cos(b_lat) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    # atan2(Y, X) 返回给定的 X 及 Y 坐标值的反正切值
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1-h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d


# 计算两个点的方位角
"""
const y = Math.sin(λ2-λ1) * Math.cos(φ2);
const x = Math.cos(φ1)*Math.sin(φ2) -
          Math.sin(φ1)*Math.cos(φ2)*Math.cos(λ2-λ1);
const θ = Math.atan2(y, x);
const brng = (θ*180/Math.PI + 360) % 360; // in degrees
"""
def bearing(a, b):
    pt_a_lat_rad = math.radians(a.lat)
    pt_a_lng_rad = math.radians(a.lng)
    pt_b_lat_rad = math.radians(b.lat)
    pt_b_lng_rad = math.radians(b.lng)
    y = math.sin(pt_b_lng_rad - pt_a_lng_rad) * math.cos(pt_b_lat_rad)
    x = math.cos(pt_a_lat_rad) * math.sin(pt_b_lat_rad) - math.sin(pt_a_lat_rad) * math.cos(pt_b_lat_rad) * math.cos(pt_b_lng_rad - pt_a_lng_rad)
    bearing_rad = math.atan2(y, x)
    # math.fmod 浮点数求模
    return math.fmod(math.degrees(bearing_rad) + 360.0, 360.0)


# 通过移动率计算坐标点位置，a,b两点确定一条路
def cal_loc_along_line(a, b, rate):
    lat = a.lat + rate * (b.lat - a.lat)
    lng = a.lng + rate * (b.lng - a.lng)
    return SPoint(lat, lng)


# 将原始点投影到一个路段中
# a,b表示路段的起始点和终点，t表示需要投影的原始点
def project_pt_to_segment(a, b, t):
    ab_angle = bearing(a, b)
    at_angle = bearing(a, t)
    ab_length = distance(a, b)
    at_length = distance(a, t)
    delta_angle = at_angle - ab_angle
    meters_along = at_length * math.cos(math.radians(delta_angle))
    if ab_length == 0.0:
        rate = 0.0
    else:
        rate = meters_along / ab_length
    if rate >= 1:
        projection = SPoint(b.lat, b.lng)
        rate = 1.0
    elif rate <= 0:
        projection = SPoint(a.lat, a.lng)
        rate = 0.0
    else:
        projection = cal_loc_along_line(a, b, rate)
    dist = distance(t, projection)

    # projection表示投影到路段上的点， rate是移动率， dist表示原始点和投影点的距离
    return projection, rate, dist



