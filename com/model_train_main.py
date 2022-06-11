import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd

import torch
import torch.optim as optim

from utils.utils import save_json_data, create_dir, load_pkl_data
from common.mbr import MBR
from common.spatial_func import SPoint, distance
from common.road_network import load_rn_shp

from models.datasets import Dataset, collate_fn, split_data
from models.model_utils import load_rn_dict, load_rid_freqs, get_rid_grid, get_poi_info, get_rn_info
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict
from models.multi_train import evaluate, init_weights, train
from models.models_attn_tandem import Encoder, DecoderMulti, Seq2SeqMulti


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()
    args_dict = {
        'module_type': 'simple', #模型类型
        'debug': False,      # True表示用少量（前3个txt）轨迹来做，False表示用全部轨迹来做
        'device': device,    # GPU or CPU

         # pre train 训练
        'load_pretrained_flag': False, #是否加载原来训练好的模型
        'model_old_path': 'D:/py/a_practice/Map_matching_project/results/simple_kr_0.25_debug_False_gs_50_lam_10_attn_True_prob_False_fea_False_20220515_121945/',   #训练好的模型的路径
        'train_flag': True,     # 训练标志
        'test_flag': True,      # 测试标志

        # attention 注意力机制
        'attn_flag': True,

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
        'split_flag': True,
        # 洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。
        'shuffle': True, # 加载数据集用到的

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

    # 准备数据，将数据进行分割 分成训练集、验证集、测试集
    print('Preparing data...')
    # 是否分割轨迹
    if args.split_flag:
        traj_input_dir = "./data/model/raw_trajectory/"
        output_dir = "./data/model/model_data/"
        split_data(traj_input_dir, output_dir)

    # 目录文件地址
    extra_info_dir = "./data/model/"
    rn_dir = "./data/map/porto_network_small/"
    train_trajs_dir = "./data/model/model_data/train_data/"
    valid_trajs_dir = "./data/model/model_data/valid_data/"
    test_trajs_dir = "./data/model/model_data/test_data/"


    # 是否使用已经训练好的模型
    if args.load_pretrained_flag:
            model_save_path = args.model_old_path
    # 重新训练，创建新的结果目录 fea_flag = False dis_prob_mask_flag = False attn_flag = True
    else:
        model_save_path = './results/'+args.module_type+'_kr_'+str(args.keep_ratio)+'_debug_'+str(args.debug)+\
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
    train_dataset = Dataset(train_trajs_dir, mbr=mbr,  weather_dict=weather_dict, parameters=args, rid_idx = rid_idx, debug=args.debug)
    valid_dataset = Dataset(valid_trajs_dir, mbr=mbr, weather_dict=weather_dict, parameters=args, rid_idx = rid_idx, debug=args.debug)
    test_dataset = Dataset(test_trajs_dir, mbr=mbr, weather_dict=weather_dict, parameters=args, rid_idx = rid_idx, debug=args.debug)


    # 读取数据集
    # pin_memory：内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。
    # collate_fn: 将一小段数据合并成数据列表 定义获取数据迭代器时，加载一个batch的数据格式
    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=collate_fn, num_workers=0)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=collate_fn, num_workers=0)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=collate_fn, num_workers=0)

    # for batch in test_iterator:
    #     src_grid_seqs, src_gps_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths = batch
    #     print('src_grid_seqs:\n',src_grid_seqs,'\nsrc_gps_seqs\n',src_gps_seqs, '\nsrc_pro_feas\n', src_pro_feas, '\nsrc_lengths\n',src_lengths, '\ntrg_gps_seqs\n', trg_gps_seqs, '\ntrg_rids\n', trg_rids, '\ntrg_rates\n', trg_rates, '\ntrg_lengths\n', trg_lengths)



    # 日志记录数据集大小
    logging.info('Finish data preparing.')
    logging.info('training dataset shape: ' + str(len(train_dataset)))
    logging.info('validation dataset shape: ' + str(len(valid_dataset)))
    logging.info('test dataset shape: ' + str(len(test_dataset)))

    # 模型定义
    enc = Encoder(args)      # 编码器
    dec = DecoderMulti(args) # 多任务的解码器模型
    model = Seq2SeqMulti(enc, dec, device).to(device) #序列到序列模型
    model.apply(init_weights)  #  初始化权重

    # 是否使用已经训练过的模型
    if args.load_pretrained_flag:
        # model.load_state_dict(torch.load(args.model_old_path + 'val-best-model.pt'))
        model.load_state_dict(torch.load(args.model_old_path + 'train-mid-model.pt'))


    # 输出和日志记录模型
    logging.info('model' + str(model))



    # 训练
    if args.train_flag:
        ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], []
        ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], []
        ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []

        dict_train_loss = {}
        dict_valid_loss = {}
        best_valid_loss = float('inf')  # compare id loss

        # get all parameters (model parameters + task dependent log variances)
        log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        for epoch in tqdm(range(args.n_epochs)):
            start_time = time.time()

            # new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, train_rate_loss, train_id_loss, result_seq = train(model, train_iterator, optimizer, log_vars,rn_dict, grid_rn_dict,  rn, raw2new_rid_dict, online_features_dict, rid_features_dict, args)
            new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, train_rate_loss, train_id_loss, result_seq = train(model, train_iterator, optimizer, log_vars, rn_dict, grid_rn_dict, rn, online_features_dict, rid_features_dict, args, rid_list)

            # valid_id_acc1, valid_id_recall, valid_id_precision, valid_dis_mae_loss, valid_dis_rmse_loss, valid_dis_rn_mae_loss, valid_dis_rn_rmse_loss, valid_rate_loss, valid_id_loss, output_seqs = evaluate(model, valid_iterator, rn_dict, grid_rn_dict, rn, raw2new_rid_dict, online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, args)

            valid_id_acc1, valid_id_recall, valid_id_precision, valid_dis_mae_loss, valid_dis_rmse_loss, valid_dis_rn_mae_loss, valid_dis_rn_rmse_loss, valid_rate_loss, valid_id_loss, output_seqs, src = evaluate(model, valid_iterator, rn_dict, grid_rn_dict, rn, online_features_dict, rid_features_dict, raw_rn_dict, args, rid_list)

            ls_train_loss.append(train_loss)
            ls_train_id_acc1.append(train_id_acc1)
            ls_train_id_recall.append(train_id_recall)
            ls_train_id_precision.append(train_id_precision)
            ls_train_rate_loss.append(train_rate_loss)
            ls_train_id_loss.append(train_id_loss)

            ls_valid_id_acc1.append(valid_id_acc1)
            ls_valid_id_recall.append(valid_id_recall)
            ls_valid_id_precision.append(valid_id_precision)
            ls_valid_dis_mae_loss.append(valid_dis_mae_loss)
            ls_valid_dis_rmse_loss.append(valid_dis_rmse_loss)
            ls_valid_dis_rn_mae_loss.append(valid_dis_rn_mae_loss)
            ls_valid_dis_rn_rmse_loss.append(valid_dis_rn_rmse_loss)
            ls_valid_rate_loss.append(valid_rate_loss)
            ls_valid_id_loss.append(valid_id_loss)
            valid_loss = valid_rate_loss + valid_id_loss
            ls_valid_loss.append(valid_loss)

            dict_train_loss['train_ttl_loss'] = ls_train_loss
            dict_train_loss['train_id_acc1'] = ls_train_id_acc1
            dict_train_loss['train_id_recall'] = ls_train_id_recall
            dict_train_loss['train_id_precision'] = ls_train_id_precision
            dict_train_loss['train_rate_loss'] = ls_train_rate_loss
            dict_train_loss['train_id_loss'] = ls_train_id_loss

            dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
            dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
            dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
            dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
            dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
            dict_valid_loss['valid_dis_mae_loss'] = ls_valid_dis_mae_loss
            dict_valid_loss['valid_dis_rmse_loss'] = ls_valid_dis_rmse_loss
            dict_valid_loss['valid_dis_rn_mae_loss'] = ls_valid_dis_rn_mae_loss
            dict_valid_loss['valid_dis_rn_rmse_loss'] = ls_valid_dis_rn_rmse_loss
            dict_valid_loss['valid_id_loss'] = ls_valid_id_loss

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print('第{}个epoch完成，耗时{}分{}秒..'.format(epoch+1,epoch_mins,epoch_secs))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')
                # torch.save(model, model_save_path + 'val-best-model1.pt')

            if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
                logging.info('\nEpoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
                weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
                logging.info('log_vars:' + str(weights))
                logging.info('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Acc1:' + str(train_id_acc1) +
                             '\tTrain RID Recall:' + str(train_id_recall) +
                             '\tTrain RID Precision:' + str(train_id_precision) +
                             '\tTrain Rate Loss:' + str(train_rate_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss))
                logging.info('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Acc1:' + str(valid_id_acc1) +
                             '\tValid RID Recall:' + str(valid_id_recall) +
                             '\tValid RID Precision:' + str(valid_id_precision) +
                             '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                             '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                             '\tValid Distance RN MAE Loss:' + str(valid_dis_rn_mae_loss) +
                             '\tValid Distance RN RMSE Loss:' + str(valid_dis_rn_rmse_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss))

                torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
                # torch.save(model, model_save_path + 'train-mid-model1.pt')
                save_json_data(dict_train_loss, model_save_path, "train_loss.json")
                save_json_data(dict_valid_loss, model_save_path, "valid_loss.json")
                torch.save(output_seqs, model_save_path + 'test_result.pth')

"""
    if args.test_flag:
        model.load_state_dict(torch.load(model_save_path + 'val-best-model.pt'))
        start_time = time.time()

        # test_id_acc1, test_id_recall, test_id_precision, test_dis_mae_loss, test_dis_rmse_loss, test_dis_rn_mae_loss, test_dis_rn_rmse_loss, test_rate_loss, test_id_loss, test_seqs = evaluate(model, test_iterator, rn_dict, grid_rn_dict, rn, raw2new_rid_dict, online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, args)

        test_id_acc1, test_id_recall, test_id_precision, test_dis_mae_loss, test_dis_rmse_loss, test_dis_rn_mae_loss, test_dis_rn_rmse_loss, test_rate_loss, test_id_loss, test_seqs = evaluate(model, test_iterator, rn_dict, grid_rn_dict, rn,  online_features_dict, rid_features_dict, raw_rn_dict, args, rid_list)

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

        # print(test_seqs)

        torch.save(test_seqs, model_save_path+'test_result.pth')

"""


