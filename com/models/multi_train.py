import numpy as np
import random

import torch
import torch.nn as nn

from models.model_utils import toseq, get_constraint_mask
from models.loss_fn import cal_id_acc, check_rn_dis_loss

# 随机数种子
SEED = 20202020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化权重
def init_weights(self):
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)

# 训练
def train(model, iterator, optimizer, log_vars, rn_dict, grid_rn_dict, rn, online_features_dict, rid_features_dict, parameters, rid_list):
    model.train()

    get_rid,get_rate = None,None    #存匹配后的结果

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()

    epoch_ttl_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    # enumerate在字典上是枚举、列举的意思 i是索引，batch是iterator的元素
    for i, batch in enumerate(iterator):

        # print('train...', i)

        # 未匹配的序列网格id+t，未匹配gps序列，特征，未匹配的除了pad的长度序列，真实的gps序列，路段id序列，移动率序列，匹配后的长度序列
        src_grid_seqs, src_gps_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths = batch

        # false 获得约束掩码层的矩阵
        if parameters.dis_prob_mask_flag:
            constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                        trg_lengths, grid_rn_dict, rn,
                                                                        parameters)
            constraint_mat = constraint_mat.permute(1, 0, 2).to(device)
            pre_grids = pre_grids.permute(1, 0, 2).to(device)
            next_grids = next_grids.permute(1, 0, 2).to(device)
        else:
            # 获取匹配后的最大长度
            max_trg_len = max(trg_lengths)
            # 批量大小
            batch_size = src_grid_seqs.size(0)
            # 掩码初始化为0
            constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size, device=device)
            # 上一个网格匹配点和下一个匹配网格设成0
            pre_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
            next_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)

        # 特征 是0。。
        src_pro_feas = src_pro_feas.float().to(device)

        src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
        trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
        trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
        trg_rates = trg_rates.permute(1, 0, 2).to(device)

        # constraint_mat = [trg len, batch size, id size]
        # src_grid_seqs = [src len, batch size, 2]
        # src_lengths = [batch size]
        # trg_gps_seqs = [trg len, batch size, 2]
        # trg_rids = [trg len, batch size, 1]
        # trg_rates = [trg len, batch size, 1]
        # trg_lengths = [batch size]

        optimizer.zero_grad()
        output_ids, output_rates = model(src_grid_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                         pre_grids, next_grids, constraint_mat, src_pro_feas,
                                         online_features_dict, rid_features_dict, parameters.tf_ratio)
        # output_ids : seq len, batch_size, rid size
        # output_rate: # seq len, batch_size, rid 1

        # squeeze 去掉维度为1的
        output_rates = output_rates.squeeze(2)
        trg_rids = trg_rids.squeeze(2)
        trg_rates = trg_rates.squeeze(2)

        ##############################################################################
        # print("#####################################################")
        # print("train的output_ids:")
        # print(output_ids.shape) # seq len,batch size, 162540(id size)
        # print("train的output_rates:")
        # print(output_rates.shape) # seq len, batch size
        #
        # print("trg的output_ids:")
        # print(trg_rids.shape) # seq len, batch size
        # print("trg的output_rates:")
        # print(trg_rates.shape) # seq len, batch size


        # print(toseq(rn_dict, output_ids, output_rates, parameters))
        # print('output rate:', output_rates.shape, ' output ids:', output_ids.shape)

        get_rid, get_rate = output_ids, output_rates

        # print("#####################################################")
        ##############################################################################

        # output_ids = [trg len, batch size, id one hot output dim]
        # output_rates = [trg len, batch size]
        # trg_rids = [trg len, batch size]
        # trg_rates = [trg len, batch size]

        # rid loss, only show and not bbp
        loss_ids1, recall, precision = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)

        # for bbp
        output_ids_dim = output_ids.shape[-1]
        output_ids = output_ids[1:].reshape(-1, output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
        trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
        # view size is not compatible with input tensor's size and stride ==> use reshape() instead


        loss_train_ids = criterion_ce(output_ids, trg_rids)
        loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
        ttl_loss = loss_train_ids + loss_rates

        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()

        epoch_ttl_loss += ttl_loss.item()
        epoch_id1_loss += loss_ids1
        epoch_recall_loss += recall
        epoch_precision_loss += precision
        epoch_train_id_loss = loss_train_ids.item()
        epoch_rate_loss += loss_rates.item()

    # print("#####################################################")

    # print('output rate:',output_rates.shape,' output ids:',output_ids.shape)

    result_seq = toseq(rn_dict, get_rid, get_rate, parameters, rid_list)

    # print(toseq(rn_dict, get_rid, get_rate, parameters))

    # print("#####################################################")

    return log_vars, epoch_ttl_loss / len(iterator), epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
           epoch_precision_loss / len(iterator), epoch_rate_loss / len(iterator), epoch_train_id_loss / len(iterator), result_seq


def evaluate(model, iterator, rn_dict, grid_rn_dict, rn,
             online_features_dict, rid_features_dict, raw_rn_dict, parameters, rid_list):
    model.eval()

    epoch_dis_mae_loss = 0
    epoch_dis_rmse_loss = 0
    epoch_dis_rn_mae_loss = 0
    epoch_dis_rn_rmse_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_rate_loss = 0
    epoch_train_id_loss = 0
    criterion_ce = nn.NLLLoss()
    criterion_reg = nn.MSELoss()

    src = []
    output = []

    with torch.no_grad():  # this line can help speed up evaluation
        for i, batch in enumerate(iterator):

            # batch一组数据
            # 未匹配的低采样轨迹网格id+在匹配后位置的序列，未匹配gps序列，特征序列（[0]*30），ground truth的gps序列，ground truth的路段id序列，ground truth的移动率序列
            src_grid_seqs, src_gps_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, src_gps, src_time = batch

            # print(src_gps)

            # 保存重采样的原始数据
            src.append({'src_gps_seqs':src_gps, 'src_time_seqs':src_time})

            # false
            if parameters.dis_prob_mask_flag:

                constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                            trg_lengths, grid_rn_dict, rn,
                                                                             parameters)
                constraint_mat = constraint_mat.permute(1, 0, 2).to(device)
                pre_grids = pre_grids.permute(1, 0, 2).to(device)
                next_grids = next_grids.permute(1, 0, 2).to(device)

            else:
                max_trg_len = max(trg_lengths)
                batch_size = src_grid_seqs.size(0)
                constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size).to(device)
                pre_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
                next_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)

            src_pro_feas = src_pro_feas.float().to(device)

            src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)

            # constraint_mat = [trg len, batch size, id size]
            # src_grid_seqs = [src len, batch size, 2]
            # src_pro_feas = [batch size, feature dim]
            # src_lengths = [batch size]
            # trg_gps_seqs = [trg len, batch size, 2]
            # trg_rids = [trg len, batch size, 1]
            # trg_rates = [trg len, batch size, 1]
            # trg_lengths = [batch size]

            output_ids, output_rates = model(src_grid_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                             pre_grids, next_grids, constraint_mat,
                                             src_pro_feas, online_features_dict, rid_features_dict,
                                             teacher_forcing_ratio=0)

            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn_dict, output_ids, output_rates, parameters, rid_list)
            output.append({'rid':output_ids, 'rate':output_rates, 'seq':output_seqs})
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)


            # output_ids = [trg len, batch size, id one hot output dim]
            # output_rates = [trg len, batch size]
            # trg_rids = [trg len, batch size]
            # trg_rates = [trg len, batch size]

            # rid loss, only show and not bbp
            loss_ids1, recall, precision = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)
            # distance loss
            # dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = check_rn_dis_loss(output_seqs[1:],
            #                                                                                    output_ids[1:],
            #                                                                                    output_rates[1:],
            #                                                                                    trg_gps_seqs[1:],
            #                                                                                    trg_rids[1:],
            #                                                                                    trg_rates[1:],
            #                                                                                    trg_lengths,
            #                                                                                    rn, raw_rn_dict,
            #                                                                                    new2raw_rid_dict)
            dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = check_rn_dis_loss(output_seqs[1:], output_ids[1:], output_rates[1:], trg_gps_seqs[1:], trg_rids[1:], trg_rates[1:], trg_lengths, rn, raw_rn_dict, rid_list)

            # for bbp
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
            loss_train_ids = criterion_ce(output_ids, trg_rids)
            # rate loss
            loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
            # loss_rates.size = [(trg len - 1), batch size], --> [(trg len - 1)* batch size,1]

            ##############################################################################
            # print("#####################################################")
            # print(trg_rids.shape)
            # print("evaluate的trg_rids:")
            # print("-----------------------------")
            # print(trg_rates.shape)
            # print("evaluate的trg_rates:")
            # # print("-----------------------------")
            # # print(output_seqs.shape)
            # # print("evaluate的output_seqs:")
            #
            # print("#####################################################")
            ##############################################################################

            epoch_dis_mae_loss += dis_mae_loss
            epoch_dis_rmse_loss += dis_rmse_loss
            epoch_dis_rn_mae_loss += dis_rn_mae_loss
            epoch_dis_rn_rmse_loss += dis_rn_rmse_loss
            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_rate_loss += loss_rates.item()
            epoch_train_id_loss += loss_train_ids.item()

        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), \
               epoch_dis_mae_loss / len(iterator), epoch_dis_rmse_loss / len(iterator), \
               epoch_dis_rn_mae_loss / len(iterator), epoch_dis_rn_rmse_loss / len(iterator), \
               epoch_rate_loss / len(iterator), epoch_train_id_loss / len(iterator), output, src
