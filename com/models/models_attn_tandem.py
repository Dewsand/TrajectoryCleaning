import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import get_dict_info_batch


# 约束掩码层 该层用于产生最大概率的eid，也就是查询到对应点最有可能落在的路段上
def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        output_custom = torch.log(x_exp / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom

class Extra_MLP(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.pro_input_dim = parameters.pro_input_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.fc_out = nn.Linear(self.pro_input_dim, self.pro_output_dim)

    def forward(self, x):
        out = torch.tanh(self.fc_out(x))
        return out

# 编码器
class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        # 隐藏状态维度 512
        self.hid_dim = parameters.hid_dim

        # 特征的输出维度 8
        self.pro_output_dim = parameters.pro_output_dim
        # 特征flag false
        self.online_features_flag = parameters.online_features_flag
        # 特征 false
        self.pro_features_flag = parameters.pro_features_flag

        # 输入维度 3 应该是 x,y,t 网格xy和对应的匹配序列位置t？
        input_dim = 3

        # # false
        # if self.online_features_flag:
        #     input_dim = input_dim + parameters.online_dim

        # 用gru
        self.rnn = nn.GRU(input_dim, self.hid_dim)
        # 丢弃信息 0.1
        self.dropout = nn.Dropout(parameters.dropout)

        # # false
        # if self.pro_features_flag:
        #     self.extra = Extra_MLP(parameters)
        #     self.fc_hid = nn.Linear(self.hid_dim + self.pro_output_dim, self.hid_dim)

    def forward(self, src, src_len, pro_features):
        # src = [src len, batch size, 3]
        # if only input trajectory, input dim = 2; elif input trajectory + behavior feature, input dim = 2 + n
        # src_len = [batch size]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer

        # hidden = [1, batch size, hidden_dim]
        # outputs = [src len, batch size, hidden_dim * num directions]
            
        # if self.pro_features_flag:
        #     extra_emb = self.extra(pro_features)
        #     extra_emb = extra_emb.unsqueeze(0)
        #     # extra_emb = [1, batch size, extra output dim]
        #     hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=2)))
        #     # hidden = [1, batch size, hid dim]

        return outputs, hidden

# 注意力层
class Attention(nn.Module):
    # TODO update to more advanced attention layer.
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e10)


        return F.softmax(attention, dim=1)

# 解码器，解码路段id和移动率
class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()

        # 路段id的大小 231805+1
        self.id_size = parameters.id_size
        # id嵌入维度 128
        self.id_emb_dim = parameters.id_emb_dim
        # 隐藏状态维度 512
        self.hid_dim = parameters.hid_dim

        # 特征输出维度 8
        self.pro_output_dim = parameters.pro_output_dim
        # 特征维度 5+5
        self.online_dim = parameters.online_dim
        # 特征维度 8
        self.rid_fea_dim = parameters.rid_fea_dim

        # 使用注意力标志 以下都是 false
        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        # id嵌入
        self.emb_id = nn.Embedding(self.id_size, self.id_emb_dim)

        # 输入维度为嵌入的维度+1
        rnn_input_dim = self.id_emb_dim + 1
        # 路段id和rate输出维度都是隐藏状态维度 512
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim

        # 输入维度为id嵌入维度+隐藏状态维度
        type_input_dim = self.id_emb_dim + self.hid_dim

        # 进入注意力模型之前的连接层 降低维度
        self.tandem_fc = nn.Sequential(
                          nn.Linear(type_input_dim, self.hid_dim),
                          nn.ReLU()
                          )

        # 是否使用注意力机制
        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim 

        # 特征使用 false
        if self.online_features_flag:
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network
        # 特征使用 false
        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim

        # 输入维度为嵌入id的维度128+1 隐藏状态维度 512
        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        # id的输出 维度为id的大小 隐藏状态维度512->231805+1
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        # 移动率的输出 隐藏状态维度512->维度为1
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        # 丢弃 0.1
        self.dropout = nn.Dropout(parameters.dropout)
        
        
    def forward(self, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                pre_grid, next_grid, constraint_vec, pro_features, online_features, rid_features):

        # input_id = [batch size, 1] rid long
        # input_rate = [batch size, 1] rate float. 
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        # attn_mask = [batch size, src len]
        # pre_grid = [batch size, 3]
        # next_grid = [batch size, 3]
        # constraint_vec = [batch size, id_size], [id_size] is the vector of reachable rid
        # pro_features = [batch size, profile features input dim]
        # online_features = [batch size, online features dim]
        # rid_features = [batch size, rid features dim]

        input_id = input_id.squeeze(1).unsqueeze(0)  # cannot use squeeze() bug for batch size = 1
        # input_id = [1, batch size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch size, 1]
        embedded = self.dropout(self.emb_id(input_id))
        # embedded = [1, batch size, emb dim]

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate, 
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                # 直接连接 嵌入的id和移动率
                rnn_input = torch.cat((embedded, input_rate), dim=2)

        # 输出 隐藏状态
        output, hidden = self.rnn(rnn_input, hidden)
        
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        assert (output == hidden).all()

        # pre_rid
        # false 约束掩码层
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(output.squeeze(0)), 
                                             constraint_vec, log_flag=True)
        else:
            # 预测的id linear层输出
            prediction_id = F.log_softmax(self.fc_id_out(output.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(self.emb_id(max_id))
        # 连接
        rate_input = torch.cat((id_emb, hidden.squeeze(0)),dim=1)
        # linear+relu
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        # false
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            # 预测的rate
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]

        return prediction_id, prediction_rate, hidden

# 序列到序列框架，集合了编码器和解码器，然后整合其他模块
class Seq2SeqMulti(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder  # Encoder
        self.decoder = decoder  # DecoderMulti
        self.device = device

    def forward(self, src, src_len, trg_id, trg_rate, trg_len,
                pre_grids, next_grids, constraint_mat, pro_features, 
                online_features_dict, rid_features_dict,
                teacher_forcing_ratio=0.5):
        """
        src = [src len, batch size, 3], x,y,t
        src_len = [batch size]
        trg_id = [trg len, batch size, 1]
        trg_rate = [trg len, batch size, 1]
        trg_len = [batch size]
        pre_grids = [trg len, batch size, 3]
        nex_grids = [trg len, batch size, 3]
        constraint_mat = [trg len, batch size, id_size]
        pro_features = [batch size, profile features input dim]
        online_features_dict = {rid: online_features} # rid --> grid --> online features
        rid_features_dict = {rid: rn_features}
        outputs_id: [seq len, batch size, id_size(1)] based on beam search
        outputs_rate: [seq len, batch size, 1]
        """
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)


        encoder_outputs, hiddens = self.encoder(src, src_len, pro_features)


        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batch_size, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None


        outputs_id, outputs_rate = self.normal_step(max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                                                    encoder_outputs, hiddens, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict,
                                                    pre_grids, next_grids, constraint_mat, pro_features,
                                                    teacher_forcing_ratio)

        return outputs_id, outputs_rate

    def normal_step(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    pre_grids, next_grids, constraint_mat, pro_features, teacher_forcing_ratio):
        """
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """

        # 输出的路段id和移动率序列 先全都初始化为0
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)


        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]

        for t in range(1, max_trg_len):

            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None

            # 预测
            prediction_id, prediction_rate, hidden = self.decoder(input_id, input_rate, hidden, encoder_outputs,
                                                                     attn_mask, pre_grids[t], next_grids[t],
                                                                     constraint_mat[t], pro_features, online_features,
                                                                     rid_features)


            # 保存输出结果
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate


            teacher_force = random.random() < teacher_forcing_ratio


            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)

            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate

        # 转化维度表示信息
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1

        # 输出结果的一个padding
        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = 0
            outputs_id[i][trg_len[i]:, 0] = 1  # make sure argmax will return eid0
            outputs_rate[i][trg_len[i]:] = 0
        # 转换维度
        outputs_id = outputs_id.permute(1, 0, 2) # seq len, batch_size, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2) # seq len, batch_size, 1

        return outputs_id, outputs_rate

