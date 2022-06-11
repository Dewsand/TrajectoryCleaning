import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd

import torch
import torch.optim as optim

from utils.utils import save_json_data, create_dir, load_pkl_data
from utils.save_traj import SaveTraj2Raw
from common.mbr import MBR
from common.spatial_func import SPoint, distance
from common.road_network import load_rn_shp
from common.trajectory import STPoint, Trajectory

from models.datasets import Dataset, collate_fn, split_data
from models.model_utils import load_rn_dict, load_rid_freqs, get_rid_grid, get_poi_info, get_rn_info
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict
from models.multi_train import evaluate, init_weights, train
from models.models_attn_tandem import Encoder, DecoderMulti, Seq2SeqMulti

from utils.parse_traj import ParseMMTraj

import datetime

# 模型地址， 测试数据地址， 匹配结果保存地址
def muti_seq2seq_match(model_path = 'D:/py/a_practice/Map_matching_project/results/simple_kr_0.25_debug_True_gs_50_lam_10_attn_False_prob_False_fea_False_20220514_121256/',test_trajs_dir = "D:/flask/TrajectoryCleaning/com/data/model/model_data/test_data/", save_dir = 'D:/flask/TrajectoryCleaning/static/download/test1/'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()
    args_dict = {
        'module_type': 'simple', #模型类型
        'debug': False,      # True表示用少量（前3个txt）轨迹来做，False表示用全部轨迹来做
        'device': device,    # GPU or CPU

         # pre train 训练
        'load_pretrained_flag': False, #是否加载原来训练好的模型
        'model_old_path': '',   #训练好的模型的路径
        'train_flag': True,     # 训练标志
        'test_flag': True,      # 测试标志

        # attention 注意力机制
        'attn_flag': False,

        # constranit 控制掩码层
        'dis_prob_mask_flag': False, #掩码标志
        'search_dist': 50,       # 掩码搜索距离
        'beta': 15,  # 参数

        # features 是否使用属性模块特征的标志
        'tandem_fea_flag': False,
        'pro_features_flag': False,
        'online_features_flag': False,

        # extra info module  属性模块添加的特征 fe和fs
        'rid_fea_dim':8,
        'pro_input_dim':30, # 24[hour] + 5[waether] + 1[holiday]
        'pro_output_dim':8,
        'poi_num':5,
        'online_dim':5+5,  # poi/roadnetwork features dim
        'poi_type':'company,food,shopping,viewpoint,house',

        # m = MBR(41.1474, -8.620, 41.1620, -8.600)
        # MBR 路网的边界 用于构建路网字典的
        'min_lat':41.1474,
        'min_lng':-8.620,
        'max_lat':41.1620,
        'max_lng':-8.60,

        # input data params 输入数据的参数
        'keep_ratio': 0.25,
        'grid_size': 50,
        'time_span': 15,  # 采样率 15秒
        'win_size': 25, # 高采样率轨迹的窗口大小？？？ 应该所 num_steps
        'ds_type': 'uniform', # 降低采样率的方法，是随机还是规律降低 ['uniform', 'random']
        'split_flag': False,
        # 洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。
        'shuffle': False, # 加载数据集用到的

        # model params 模型参数
        'hid_dim': 512, # 隐藏状态的长度
        'id_emb_dim': 128,   # id嵌入的长度
        'dropout': 0.1,      # 丢弃法的丢弃率
        # 'id_size':2571+1,   # id的大小 rn_dict 的id大小
        'id_size': 2349 + 1,  # id的大小 rn_dict 的id大小 其实边的数目应该是 162539 这里可以弄一个边的映射字典

        'lambda1': 10,  # 多任务的损失权重
        'n_epochs': 200,  # epoch
        'batch_size': 64,       # 批量大小
        # 'learning_rate':1e-3,   # 学习率
        'learning_rate': 1e-3,  # 学习率 学习率改大一点?，收敛快一点. 不能改？？？？
        'tf_ratio': 0.5,         # 不知道 强制教学的概率 teacher_forcing_ratio
        'clip': 1,               # 剪枝？？
        'log_step': 1            # 日志相关的
    }

    # 更新参数
    args.update(args_dict)

    # 目录文件地址
    extra_info_dir = "com/data/model/"
    rn_dir = "com/data/map/porto_network_small/"

    model_save_path = './results/test_'+args.module_type+'_kr_'+str(args.keep_ratio)+'_debug_'+str(args.debug)+\
    '_gs_'+str(args.grid_size)+'_lam_'+str(args.lambda1)+\
    '_attn_'+str(args.attn_flag)+'_prob_'+str(args.dis_prob_mask_flag)+\
    '_fea_'+str(False)+'_'+time.strftime("%Y%m%d_%H%M%S") + '/'

    # 创建结果的存放目录
    create_dir(model_save_path)

    # 记录训练日志信息
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', filename=model_save_path + 'log.txt', filemode='a')

    # 加载路网
    rn = load_rn_shp(rn_dir, is_directed=True)

    # 加载路网字典集（路段对应的起点终点和长度）
    raw_rn_dict = load_rn_dict(extra_info_dir, file_name='porto_rn_dict_small.json')

    #加载较少路段的字典集 已修改成raw
    rn_dict = raw_rn_dict

    # 路段id列表 rid_list[idx-1]得到对应的真实路段id
    rid_list = list(rn_dict.keys())
    rid_idx = {}  # 路段id的索引列表
    for i in range(len(rid_list)):
        # 从1开始表示
        rid_idx[rid_list[i]] = i+1

    # 栅格表示的路网 以及每个网格对应的道路id 最大索引下标以及最小索引下标
    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
    args_dict['max_xid'] = max_xid
    args_dict['max_yid'] = max_yid

    # 更新参数
    args.update(args_dict)

    # 日志记录参数信息
    logging.info(args_dict)

    # 时间和假期等信息 不使用
    weather_dict = None

    # 其他道路特征信息也设置为 None 表示不使用
    norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None

    # 路网特征设置为 None
    rid_features_dict = None

    # load dataset 加载数据集 训练集、验证集、测试集
    test_dataset = Dataset(test_trajs_dir, mbr=mbr, weather_dict=weather_dict, parameters=args, rid_idx = rid_idx, debug=args.debug)


    # 读取数据集
    # pin_memory：内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。
    # collate_fn: 将一小段数据合并成数据列表 定义获取数据迭代器时，加载一个batch的数据格式
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=collate_fn, num_workers=0)


    # 日志记录数据集大小
    logging.info('Finish data preparing.')
    logging.info('test dataset shape: ' + str(len(test_dataset)))

    # 模型定义
    enc = Encoder(args)      # 编码器
    dec = DecoderMulti(args) # 多任务的解码器模型
    model = Seq2SeqMulti(enc, dec, device).to(device) #序列到序列模型
    model.apply(init_weights)  #  初始化权重


    # 输出和日志记录模型
    logging.info('model' + str(model))



    if args.test_flag:
        model.load_state_dict(torch.load(model_path + 'train-mid-model.pt'))
        start_time = time.time()

        test_id_acc1, test_id_recall, test_id_precision, test_dis_mae_loss, test_dis_rmse_loss, test_dis_rn_mae_loss, test_dis_rn_rmse_loss, test_rate_loss, test_id_loss, test_seqs, src = evaluate(model, test_iterator, rn_dict, grid_rn_dict, rn,  online_features_dict, rid_features_dict, raw_rn_dict, args, rid_list)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logging.info('\n\nTest Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
        logging.info('\tTest RID Acc1:' + str(test_id_acc1) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                     '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                     '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                     '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                     '\tTest Rate Loss:' + str(test_rate_loss) +
                     '\tTest RID Loss:' + str(test_id_loss))

        # 匹配轨迹，原始轨迹, 原始轨迹时间序列
        match_seq = []
        src_seq = []
        src_time = []

        # 将结果序列剔除掉一些填充和开始元素
        for seq in test_seqs:
            clean_seqs = []
            seqs = seq['seq'].permute(1, 0, 2).numpy().tolist()
            for s in seqs:
                n_s = []
                for item in s:
                    if item != [0.0, 0.0]:
                       n_s.append(item)
                clean_seqs.append(n_s)
            match_seq.extend(clean_seqs)


        for seq in src:
            src_seq.extend(seq['src_gps_seqs'])
            src_time.extend(seq['src_time_seqs'])



    raw_traj = []

    # 分割后的测试原始轨迹
    for i in range(len(src_seq)):
        oid = 'o_' + str(i)
        tid = 't_' + str(i)
        ptlist = []
        seq = src_seq[i]
        for j in range(len(seq)):
            ptlist.append(STPoint(lat=seq[j][0], lng=seq[j][1], time=src_time[i][j]))
        raw_traj.append(Trajectory(oid=oid, tid=tid, pt_list=ptlist))

    match_traj = []
    # 匹配后的轨迹生成
    for i in range(len(match_seq)):
        oid = 'o_' + str(i)
        tid = 't_' + str(i)
        ptlist = []
        seq = match_seq[i]
        t = src_time[i][0]
        for j in range(len(seq)):
            ptlist.append(STPoint(lat=seq[j][0], lng=seq[j][1], time=t+datetime.timedelta(seconds=j*15)))
        match_traj.append(Trajectory(oid=oid, tid=tid, pt_list=ptlist))

    save = SaveTraj2Raw('GCJ02ToWGS84')

    # dir = '../static/download/test1/'
    create_dir(save_dir)
    save.store(raw_traj, save_dir + 'raw.txt')
    save.store(match_traj, save_dir + 'match.txt')

    # 返回分割原始轨迹文件地址以及匹配文件地址
    return save_dir + 'raw.txt', save_dir + 'match.txt'



# muti_seq2seq_match()

