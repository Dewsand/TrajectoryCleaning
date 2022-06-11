import random
from tqdm import tqdm
import os
from chinese_calendar import is_holiday

import numpy as np
import torch

from common.spatial_func import distance
from common.trajectory import get_tid, Trajectory
from utils.parse_traj import ParseMMTraj
from utils.save_traj import SaveTraj2MM
from utils.utils import create_dir
from .model_utils import load_rid_freqs

# 分割数据集，将原始数据分成训练集、验证集、测试集 7：2：1
def split_data(traj_input_dir, output_dir):
    create_dir(output_dir)
    train_data_dir = output_dir + 'train_data/'
    create_dir(train_data_dir)
    val_data_dir = output_dir + 'valid_data/'
    create_dir(val_data_dir)
    test_data_dir = output_dir + 'test_data/'
    create_dir(test_data_dir)

    trg_parser = ParseMMTraj()
    trg_saver = SaveTraj2MM()

    for file_name in os.listdir(traj_input_dir):
        traj_input_path = os.path.join(traj_input_dir, file_name)
        trg_trajs = np.array(trg_parser.parse(traj_input_path))
        ttl_lens = len(trg_trajs)
        test_inds = random.sample(range(ttl_lens), int(ttl_lens * 0.05))  # 10% as test data
        tmp_inds = [ind for ind in range(ttl_lens) if ind not in test_inds]
        val_inds = random.sample(tmp_inds, int(ttl_lens * 0.05))  # 10% as validation data
        train_inds = [ind for ind in tmp_inds if ind not in val_inds]  # 80% as training data

        trg_saver.store(trg_trajs[train_inds], os.path.join(train_data_dir, 'train_' + file_name))
        print("target traj train len: ", len(trg_trajs[train_inds]))
        trg_saver.store(trg_trajs[val_inds], os.path.join(val_data_dir, 'val_' + file_name))
        print("target traj val len: ", len(trg_trajs[val_inds]))
        trg_saver.store(trg_trajs[test_inds], os.path.join(test_data_dir, 'test_' + file_name))
        print("target traj test len: ", len(trg_trajs[test_inds]))

# 定义训练的数据集，重写torch的数据集类，定义自己需要的格式
class Dataset(torch.utils.data.Dataset):

    # 轨迹的目录, 路网边界, 路网poi特征字典, 路网特征字典, 天气特征, 参数,
    def __init__(self, trajs_dir, mbr, weather_dict, parameters, rid_idx, debug=True):
        # 所有轨迹的边界 （我用的是地图的边界。。之前写了一个计算轨迹边界的，可以改一下，先用地图的边界）
        self.mbr = mbr  # MBR of all trajectories
        # 路网字典的网格大小
        self.grid_size = parameters.grid_size
        # 两个连续采样点的时间间隔
        self.time_span = parameters.time_span  # time interval between two consecutive points.
        # 特征。。。设成了false
        self.online_features_flag = parameters.online_features_flag
        # 原轨迹的网格序列, gps序列
        self.src_grid_seqs, self.src_gps_seqs = [], []
        # 特征序列（与weather有关的）
        self.src_pro_feas = []
        # 目标gps序列, 路段id序列, 移动率序列
        self.trg_gps_seqs, self.trg_rids, self.trg_rates = [], [], []
        #  切割之后的轨迹id序列
        self.new_tids = []

        # 原始轨迹时间序列
        self.src_time = []

        # above should be [num_seq, len_seq(unpadded)]
        # 加载数据 'ds_type':'random'
        self.get_data(trajs_dir, weather_dict, parameters.win_size, parameters.ds_type, parameters.keep_ratio, debug, rid_idx)



    # 数据集的长度，也就是轨迹的条数 需要通过输入的轨迹dir找到轨迹文件，加载计算轨迹数目 在get_data里面实现
    def __len__(self):
        return len(self.src_grid_seqs)

    # 获取一组数据 相当于return的数据返回给collect_fn
    def __getitem__(self, index):
        src_grid_seq = self.src_grid_seqs[index]
        src_gps_seq = self.src_gps_seqs[index]
        trg_gps_seq = self.trg_gps_seqs[index]
        trg_rid = self.trg_rids[index]
        trg_rate = self.trg_rates[index]
        src_time = self.src_time[index]

        src_gps = src_gps_seq
        # print(src_gps)

        # 为序列添加开始标记
        src_grid_seq = self.add_token(src_grid_seq)
        src_gps_seq = self.add_token(src_gps_seq)
        trg_gps_seq = self.add_token(trg_gps_seq)
        trg_rid = self.add_token(trg_rid)
        trg_rate = self.add_token(trg_rate)

        # 获取对应批量序列的特征
        src_pro_fea = torch.tensor(self.src_pro_feas[index])

        # 给collect_fn转送的数据
        # 未匹配的低采样轨迹网格id+在匹配后位置的序列，未匹配gps序列，特征序列（[0]*30），ground truth的gps序列，ground truth的路段id序列，ground truth的移动率序列
        # print(src_gps_seq)
        return src_grid_seq, src_gps_seq, src_pro_fea, trg_gps_seq, trg_rid, trg_rate, src_gps, src_time

    # 添加开始标记 sos start of sequence ,将序列转为tensor向量
    def add_token(self, sequence):
        new_sequence = []
        dimension = len(sequence[0]) # 查看序列中每个元素的维度值，决定需要添加多少个0作为开始标记
        start = [0] * dimension  # 开始标记
        new_sequence.append(start)  # 添加开始标记
        new_sequence.extend(sequence) # 组合开始标志以及原序列
        new_sequence = torch.tensor(new_sequence) # 转为tensor，方便训练计算
        return new_sequence

    # 对Dataset的数据进行计算赋值等一系列初始化操作
    def get_data(self, trajs_dir, weather_dict, win_size, ds_type, keep_ratio, debug, rid_idx):

        parser = ParseMMTraj()

        # 测试版本的话，只取少量数据（前3个文件的轨迹），否则取全部轨迹进行训练
        if debug:
            # os.listdir 返回指定路径下的文件和文件夹列表
            trg_paths = os.listdir(trajs_dir)[:3]
            num = -1
        else:
            trg_paths = os.listdir(trajs_dir)
            num = -1

        # 加载轨迹信息，进行数据集属性的初始化
        for file_name in tqdm(trg_paths):
            # 获取文件中的轨迹 用的是ParseMMTraj()的普通解析
            trajs = parser.parse(os.path.join(trajs_dir, file_name))
            # 解析每一条轨迹的轨迹点信息
            for traj in trajs[:num]:
                # 解析轨迹 用dataset的解析一条轨迹的轨迹点信息
                # # 经过分割后新的轨迹id序列，匹配后的gps序列，匹配后的路段id序列，匹配后的移动率序列，未匹配的网格序列，未匹配的gps序列，特征序列（特征为[0]*30）
                new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, ls_grid_seq_ls, ls_gps_seq_ls, features_ls, time_ls = self.parse_traj(traj, weather_dict, win_size, ds_type, keep_ratio, rid_idx)

                # 添加进大的列表里面 为Dataset里面的数据赋值。。
                if new_tid_ls is not None:
                    self.new_tids.extend(new_tid_ls)
                    self.trg_gps_seqs.extend(mm_gps_seq_ls)
                    self.trg_rids.extend(mm_eids_ls)
                    self.trg_rates.extend(mm_rates_ls)
                    self.src_grid_seqs.extend(ls_grid_seq_ls)
                    self.src_gps_seqs.extend(ls_gps_seq_ls)
                    self.src_pro_feas.extend(features_ls)
                    self.src_time.extend(time_ls)
                    assert len(new_tid_ls) == len(mm_gps_seq_ls) == len(mm_eids_ls) == len(mm_rates_ls)

        assert len(self.new_tids) == len(self.trg_gps_seqs) == len(self.trg_rids) == len(self.trg_rates) == len(self.src_gps_seqs) == len(self.src_grid_seqs) == len(self.src_pro_feas), 'The number of source and target sequence must be equal.'

    # 解析一条轨迹的信息
    def parse_traj(self, traj, weather_dict, win_size, ds_type, keep_ratio, rid_idx):
        # 得到目标窗口大小的轨迹 分割轨迹
        new_trajs = self.get_win_trajs(traj, win_size)

        # print('未分割前的轨迹：', traj)
        # print('分割后的轨迹：', new_trajs)

        new_tid_ls = []
        # 匹配点的信息
        mm_gps_seq_ls, mm_eids_ls, mm_rates_ls = [], [], []
        # 未匹配的序列信息
        ls_grid_seq_ls, ls_gps_seq_ls, features_ls, time_ls = [], [], [], []


        for tr in new_trajs:
            # 解析子轨迹
            tmp_pt_list = tr.pt_list

            # 新的轨迹 轨迹id序列，因为经过分割后，轨迹长的变成短的轨迹子序列，所以有新的轨迹id序列，对应所有分割后的轨迹的数目
            new_tid_ls.append(tr.tid)

            # get target sequence 获取目标序列（匹配好的）gps序列，路段id，移动率
            mm_gps_seq, mm_eids, mm_rates = self.get_trg_seq(tmp_pt_list, rid_idx)
            if mm_eids is None:
                return None, None, None, None, None, None, None

            # get source sequence
            # 降低未匹配序列的采样率 ds_type表示随机降低？ = random
            ds_pt_list = self.downsample_traj(tmp_pt_list, ds_type, keep_ratio)

            # print('\n未降低采样率之前的list：', [l.time for l in tmp_pt_list], 'len:', len(tmp_pt_list))
            # print('降低采样率之后的ptlist：', [l.time for l in ds_pt_list], 'len:', len(ds_pt_list))

            # 获取未匹配的序列信息（经过降低采样率之后的）
            # 网格id和对应应该在的位置序列（x，y，t），未匹配的gps序列，轨迹点的小时序列， 就是匹配后一共多少点，一个数
            ls_grid_seq, ls_gps_seq, hours, ttl_t, ls_time = self.get_src_seq(ds_pt_list)

            # 特征序列 设成0了都，[0]*30
            features = self.get_pro_features(ds_pt_list, hours, weather_dict)

            # check if src and trg len equal, if not return none
            # 确保预计的匹配后长度与ground truth，也就是真实的匹配长度相等
            if len(mm_gps_seq) != ttl_t:
                print('\n匹配与预计长度不一致！！！\n')
                return None, None, None, None, None, None, None
            
            mm_gps_seq_ls.append(mm_gps_seq)
            mm_eids_ls.append(mm_eids)
            mm_rates_ls.append(mm_rates)
            ls_grid_seq_ls.append(ls_grid_seq)
            ls_gps_seq_ls.append(ls_gps_seq)
            features_ls.append(features)
            time_ls.append(ls_time)

        # print(ls_gps_seq_ls)
        # 经过分割后新的轨迹id序列，匹配后的gps序列，匹配后的路段id序列，匹配后的移动率序列，未匹配的网格序列，未匹配的gps序列，特征序列（特征为[0]*30）
        return new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, ls_grid_seq_ls, ls_gps_seq_ls, features_ls, time_ls

    # 切割轨迹，转为指定窗口大小的轨迹序列
    def get_win_trajs(self, traj, win_size):
        pt_list = traj.pt_list
        len_pt_list = len(pt_list)
        # 轨迹点数目小于win_size，直接返回该轨迹组成列表，否则切割轨迹
        if len_pt_list < win_size:
            return [traj]

        # 转换后的轨迹数量
        num_win = len_pt_list // win_size

        # 最后一个片段的轨迹的长度
        last_traj_len = len_pt_list % win_size + 1

        new_trajs = []
        for w in range(num_win):
            # if last window is large enough then split to a single trajectory
            # 最后一条分割的轨迹，轨迹点大于15个的话，则单独成立为一条轨迹，否则将这个轨迹融合到倒数第一条中
            if w == num_win and last_traj_len > 15:
                tmp_pt_list = pt_list[win_size * w - 1:]
            # elif last window is not large enough then merge to the last trajectory
            elif w == num_win - 1 and last_traj_len <= 15:
                # fix bug, when num_win = 1
                ind = 0
                if win_size * w - 1 > 0:
                    ind = win_size * w - 1
                tmp_pt_list = pt_list[ind:]
            # else split trajectories based on the window size
            else:
                tmp_pt_list = pt_list[max(0, (win_size * w - 1)):win_size * (w + 1)]
                # -1 to make sure the overlap between two trajs

            new_traj = Trajectory(traj.oid, get_tid(traj.oid, tmp_pt_list), tmp_pt_list)
            new_trajs.append(new_traj)
        return new_trajs

    # 获取分割后的子轨迹的匹配点序列信息 gps序列，路段id，移动率
    def get_trg_seq(self, tmp_pt_list, rid_idx):
        mm_gps_seq = []
        mm_eids = []
        mm_rates = []
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            if candi_pt is None:
                return None, None, None
            else:
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
                mm_eids.append([rid_idx[candi_pt.eid]])  # keep the same format as seq
                mm_rates.append([candi_pt.rate])
        return mm_gps_seq, mm_eids, mm_rates

    # 对未匹配的序列进行解析 降低采样率之后的未匹配轨迹
    def get_src_seq(self, ds_pt_list):
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        ls_time = []
        first_pt = ds_pt_list[0]
        last_pt = ds_pt_list[-1]
        time_interval = self.time_span
        # 一共有多少个点计算 （就是匹配后一共多少点）是一个数
        ttl_t = self.get_noramlized_t(first_pt, last_pt, time_interval)
        for ds_pt in ds_pt_list:
            # 获取小时信息，应该所要和weather对应的
            hours.append(ds_pt.time.hour)
            # 计算该点在匹配后应该在第几个点
            t = self.get_noramlized_t(first_pt, ds_pt, time_interval)
            # 坐标点
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            # print([ds_pt.lat, ds_pt.lng])
            # 计算坐标点网格所在
            locgrid_xid, locgrid_yid = self.gps2grid(ds_pt, self.mbr, self.grid_size)

            # t表示匹配后这个点在序列的第几个点
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])

            ls_time.append(ds_pt.time)

        # print('lslslsls\n',ls_grid_seq, ls_gps_seq, hours, ttl_t)

        # 网格id和对应应该在的位置序列（x，y，t），未匹配的gps序列，轨迹点的小时序列， 就是匹配后一共多少点，一个数
        return ls_grid_seq, ls_gps_seq, hours, ttl_t, ls_time

    # 获得特征向量 设为[0]*30了。。 不做特征表示
    def get_pro_features(self, ds_pt_list, hours, weather_dict):

        if weather_dict is None:
            return torch.zeros(30,dtype=torch.int8).numpy().tolist()

        holiday = is_holiday(ds_pt_list[0].time)*1
        day = ds_pt_list[0].time.day
        hour = {'hour': np.bincount(hours).max()}  # find most frequent hours as hour of the trajectory
        weather = {'weather': weather_dict[(day, hour['hour'])]}
        features = self.one_hot(hour) + self.one_hot(weather) + [holiday]
        return features
    
    # 计算一个点在网格化之后的哪个位置 返回位置的x和y
    def gps2grid(self, pt, mbr, grid_size):

        LAT_PER_METER = 8.993203677616966e-06
        LNG_PER_METER = 1.1700193970443768e-05
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size
        
        max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
        max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1
        
        lat = pt.lat
        lng = pt.lng
        locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
        locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1
        
        return locgrid_x, locgrid_y
    
    # 计算一个点应该在匹配后的序列中的第几个位置，用采样的时间间隔计算
    def get_noramlized_t(self, first_pt, current_pt, time_interval):
        t = int(1+((current_pt.time - first_pt.time).seconds/time_interval))
        return t

    @staticmethod
    def get_distance(pt_list):
        dist = 0.0
        pre_pt = pt_list[0]
        for pt in pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist += tmp_dist
            pre_pt = pt
        return dist

    # 降低未匹配轨迹的采样率，得到低采样率的轨迹
    @staticmethod
    def downsample_traj(pt_list, ds_type, keep_ratio):

        # uniform表示有规律地降低采样率，random表示在一个采样间隔内随机取样降低采样率
        assert ds_type in ['uniform', 'random'], 'only `uniform` or `random` is supported'

        old_pt_list = pt_list.copy()
        start_pt = old_pt_list[0]
        end_pt = old_pt_list[-1]

        if ds_type == 'uniform':
            if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)]
            else:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
        elif ds_type == 'random':
            sampled_inds = sorted(
                random.sample(range(1, len(old_pt_list) - 1), int((len(old_pt_list) - 2) * keep_ratio)))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]

        return new_pt_list


    @staticmethod
    def one_hot(data):
        one_hot_dict = {'hour': 24, 'weekday': 7, 'weather':5}
        for k, v in data.items():
            encoded_data = [0] * one_hot_dict[k]
            encoded_data[v - 1] = 1
        return encoded_data


# 载一个批量数据，自定义可能会出错，重写collate_fn函数
def collate_fn(data):
    """
    从元组列表（src_seq、src_pro_fea、trg_seq、trg_rid、trg_rate）创建小批量张量。
    我们应该构建一个自定义的 collate_fn 而不是使用默认的 collate_fn，
    因为默认不支持合并序列（包括填充）。
    序列被填充到小批量序列的最大长度（动态填充）
    src_grid_seqs:(batch_size, padded_length, 3)
    src_gps_seqs:(batch_size, padded_length, 3).
    src_pro_feas:(batch_size, feature_dim)
    src_lengths
    trg_seqs:(batch_size, padded_length, 2).
    trg_rids:(batch_size, padded_length, 1).
    trg_rates:(batch_size, padded_length, 1).
    trg_lengths
    """

    def merge(sequences):
        # 存每条序列的长度
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        # 填充 <pad> 使每条序列统一最长长度，没有设置最长长度为多少
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by source sequence length (descending order) to use pack_padded_sequence
    # 排序？找出最长的序列？然后添加padding？
    # print(data)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # print(data, '\n')

    # seperate source and target sequences 从dataset传来的信息
    # 未匹配的低采样轨迹网格id+在匹配后位置的序列，未匹配gps序列，特征序列（[0]*30），ground truth的gps序列，ground truth的路段id序列，ground truth的移动率序列
    src_grid_seqs, src_gps_seqs, src_pro_feas, trg_gps_seqs, trg_rids, trg_rates, src_gps, src_time = zip(*data)  # unzip data

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    # 合并？ 1维张量变成2维张量
    src_grid_seqs, src_lengths = merge(src_grid_seqs)
    src_gps_seqs, src_lengths = merge(src_gps_seqs)
    src_pro_feas = torch.tensor([list(src_pro_fea) for src_pro_fea in src_pro_feas])
    trg_gps_seqs, trg_lengths = merge(trg_gps_seqs)
    trg_rids, _ = merge(trg_rids)
    trg_rates, _ = merge(trg_rates)

    # 未匹配的序列网格id+t，未匹配gps序列，特征，未匹配的除了pad的长度序列，真实的gps序列，路段id序列，移动率序列，匹配后的长度序列

    # print(src_gps_seqs.numpy().tolist())
    # print(src_gps_seqs)
    return src_grid_seqs, src_gps_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, src_gps, src_time

