# DKT使用了embedding进行了复现

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,3"
from EduKTM import KTM
import logging
import torch
import torch.nn as nn
from torch.nn import Module, LSTM, Linear, Dropout,GRU
import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch.nn.functional as F
import pickle
import  numpy as np

from Findembedding import num2codeemb, num2skillemb
from sklearn.metrics import accuracy_score
from ast_tree_dist_socre import compute_edit_distance,compute_edit_distance_fast,euclidean_distance


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

with open('./data/Afterprocess/problems_codenet_python.pkl', 'rb') as f:
    data = pickle.load(f)

# edge_weight = []
def gen_edges_for_sequence(p,sequence,edge_weight):
    in_vec = []
    out_vec = []
    # edge_weight = []
    new_squence=[x for x in sequence if x != 0]
    for i in range(len(new_squence) - 1):
        if new_squence[i] == new_squence[i + 1]:  # 如果当前节点和下一个节点相同
            in_vec.append(i)
            out_vec.append(i+1)
            # w = compute_edit_distance_fast(data[p[i].item()],data[p[i+1].item()])
            w = euclidean_distance(num2codeemb(p[i].item()), num2codeemb(p[i+1].item()))
            edge_weight.append(w)
    return [in_vec, out_vec]


#无向边
# def gen_edges_for_sequence(p, sequence, edge_weight):
#     in_vec = []
#     out_vec = []
#     new_sequence = [x for x in sequence if x != 0]
#
#     for i in range(len(new_sequence) - 1):
#         if new_sequence[i] == new_sequence[i + 1]:  # 如果当前节点和下一个节点相同
#             in_vec.append(i)
#             out_vec.append(i + 1)
#             # # 添加反向边
#             # in_vec.append(i + 1)
#             # out_vec.append(i)
#
#             # 计算边的权重
#             w = euclidean_distance(num2codeemb(p[i].item()), num2codeemb(p[i + 1].item()))
#             edge_weight.append(w)
#             edge_weight.append(w)  # 反向边的权重与正向边相同
#
#     return [in_vec, out_vec]


#生成每个学生序列的同一个知识点的图  最后批量化
# def gen_edges_for_sequence(sequence):
#     in_vec = []
#     out_vec = []
#     # 计算除去0之外的元素个数
#     non_zero_sequence = [x for x in sequence if x != 0]
#     for i in range(len(non_zero_sequence)-1):
#         # 如果当前节点非0，则添加一个自环
#         if non_zero_sequence[i] != 0:
#             in_vec.append(i)  # 添加自环的入边
#             out_vec.append(i)  # 添加自环的出边
#         # 如果当前节点和下一个节点相同且非0，则添加边
#         if non_zero_sequence[i] == non_zero_sequence[i + 1]:
#             in_vec.append(i)  # 添加有向边
#             out_vec.append(i+1)
#     # 由于上面的循环不会处理最后一个元素，我们需要在循环外检查最后一个元素是否应该添加自环
#     if non_zero_sequence[-1] != 0:
#         in_vec.append(len(non_zero_sequence)-1)
#         out_vec.append(len(non_zero_sequence)-1)
#     return [in_vec, out_vec]

#连续相同的元素 只取他的第一个和最后一个进行边的联系
# def gen_edges_for_sequence(sequence):
#     in_vec = []
#     out_vec = []
#
#     # 找到最后一个非零元素的位置
#     last_non_zero_index = len(sequence) - 1
#     # while sequence[last_non_zero_index] == 0 and last_non_zero_index > 0:
#     #     last_non_zero_index -= 1
#
#     i = 0
#     while i <= last_non_zero_index:
#         if sequence[i] != 0:  # 如果当前元素不为0
#             start = i
#             while i < last_non_zero_index and sequence[i] == sequence[i + 1]:  # 当前元素和下一个元素相同
#                 i += 1
#             end = i
#             if end > start:  # 如果有连续两个以上的非零元素
#                 in_vec.append(start)  # 添加从第一个相同元素到最后一个相同元素的边
#                 out_vec.append(end)
#         i += 1
#
#     return [in_vec, out_vec]

def batch_genr_edge_index(batch_p,batch_data):
    batch_size, _ = batch_data.shape
    batch_edge_index = []
    batch_edge_weight = []
    edge_weight = []
    for i in range(batch_size):
        edge_index = gen_edges_for_sequence(batch_p[i],batch_data[i],edge_weight)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        batch_edge_index.append(edge_index)

        # edge_weight_temp = torch.tensor(edge_weight_temp, dtype=torch.long)
        # batch_edge_weight.append(edge_weight_temp)

    return batch_edge_index,edge_weight


#生成所有同一个学生所有相近知识点之间的关系图，此关系图是所有知识点之间关系questions_graph.gpickle的子图  最后批量化
with open("questions_graph_threshold_0.8.gpickle", "rb") as f:
    G = pickle.load(f)

# 遍历序列中的每个知识点，添加边到subG，只添加在G中已经存在的边
def gen_similar_edges_for_sequence(sequence):
    in_vec = []
    out_vec = []
    # 计算除去0之外的元素个数
    non_zero_sequence = [x for x in sequence if x != 0]
    for i, node in enumerate(sequence):
        # 确保自己到自己也有一条边
        if node in G.nodes:
            in_vec.append(i)
            out_vec.append(i)
        for j in range(i + 1, len(non_zero_sequence)):
            if sequence[j] in G.nodes and G.has_edge(node, sequence[j]):
                in_vec.append(i)
                out_vec.append(j)

                # # 添加反向边
                # in_vec.append(j)
                # out_vec.append(i)
    return [in_vec, out_vec]

def batch_similar_genr_edge_index(batch_data):
    batch_size, _ = batch_data.shape
    batch_edge_index = []

    for i in range(batch_size):
        edge_index = gen_similar_edges_for_sequence(batch_data[i])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        batch_edge_index.append(edge_index)

    return batch_edge_index

import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


#自定义一个GCN 起名：逻辑距离权重卷积网络  逻辑距离=树编辑距离*（质量得分之差）
class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__(aggr='add')  # "add" aggregation.
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  #添加自环
        if edge_weight is not None:
            # 创建一个新的edge_weight，其大小等于新的edge_index的大小
            edge_weight = torch.tensor(edge_weight, dtype=x.dtype, device=x.device)
            new_edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)
            # 将原始的edge_weight复制到新的edge_weight中
            new_edge_weight[:len(edge_weight)] = edge_weight

            # Standardize the weights
            # mean = torch.mean(new_edge_weight)
            # std = torch.std(new_edge_weight)
            # new_edge_weight = (new_edge_weight - mean) / std

            # Normalize the weights
            min_val = torch.min(new_edge_weight)
            max_val = torch.max(new_edge_weight)
            new_edge_weight = (new_edge_weight - min_val) / (max_val - min_val)

            # Apply log transformation
            # new_edge_weight = torch.log1p(new_edge_weight)

            # Max absolute scaling
            # max_abs_val = torch.max(torch.abs(new_edge_weight))
            # new_edge_weight = new_edge_weight / max_abs_val


        # Step 2: Compute normalization
        if edge_weight is None:
            new_edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        # 将列表转换为张量
        # 将列表的列表平铺成一个列表
        # flat_list = [item for sublist in edge_weight for item in sublist]
        # # 将平铺后的列表转换为张量
        # # edge_weight = torch.tensor(flat_list, dtype=x.dtype, device=x.device)
        # edge_weight = torch.tensor(edge_weight, dtype=x.dtype, device=x.device)
        norm = deg_inv_sqrt[row] * new_edge_weight * deg_inv_sqrt[col]

        # Step 3: Apply linear transformation and propagate through the graph.
        x = self.linear(x)

        # Step 4: Message passing.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Scale node features by the normalized edge weights.
        return norm.view(-1, 1) * x_j


import torch
from torch import nn


#transformer
# class MyModel(nn.Module):
#     def __init__(self, d_model, nhead, num_layers):
#         super().__init__()
#         self.transformer_layer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model, nhead), num_layers)
#         self.dropout_layer = nn.Dropout(0.1)
#         self.out_layer = nn.Linear(1536, 1)
#
#     def forward(self, p_emb, q_emb, similar_skill_state,same_skill_state, r_emb, q_next_emb):
#         # Concatenate inputs
#         concatenated_qr = torch.cat((p_emb, q_emb, similar_skill_state,same_skill_state, r_emb), dim=-1)
#
#         # Create attention mask
#         batch_size, seq_length = concatenated_qr.size(0), concatenated_qr.size(1)
#         mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)
#
#         # mask = torch.triu(mask, diagonal=1).bool().to(device)
#         mask = torch.tril(mask, diagonal=-1).bool().to(device)
#
#         # Pass through Transformer
#         h = self.transformer_layer(concatenated_qr.transpose(0, 1), src_key_padding_mask=mask)
#         h = h.transpose(0, 1)
#
#         # Pass through dropout layer
#         h = self.dropout_layer(h)
#
#         # Concatenate with next question embedding
#         h = torch.cat((h, q_next_emb), dim=-1)
#
#         # Pass through output layer
#         y = self.out_layer(h)
#
#         # Apply sigmoid activation
#         y = torch.sigmoid(y)
#
#         return y




seqlen = 220    #codenet-python:220
class DKT(Module):
    def __init__(self,num_p, num_q,num_g,num_cpu,num_mem,emb_size, dropout=0.03):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lstm_layer = LSTM(self.emb_size * 5, self.hidden_size, batch_first=True)
        self.lstm_layer_qc = LSTM(self.emb_size * 1 + 1, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.dropout_layer_qc = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size*2, 1 )
        self.embedding_problem = nn.Embedding(num_p+1,emb_size)
        # torch.nn.init.xavier_uniform_(self.embedding_problem.weight)
        self.embedding_question = nn.Embedding(num_q + 1, emb_size)
        # torch.nn.init.xavier_uniform_(self.embedding_question.weight)
        self.answer_embedding = nn.Embedding(2, emb_size)
        # torch.nn.init.xavier_uniform_(self.answer_embedding.weight)

        #使用transform替代lstm试一下
        # self.transformer_layer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model, 8), 6)



        # 新增GAT
        self.similar_code_concat_layer = Linear(self.hidden_size * 2, self.hidden_size)
        self.same_code_concat_layer = Linear(self.hidden_size * 2, self.hidden_size)  # New fully connected layer
        # self.conv1 = GATConv(self.hidden_size, 256, heads=8, dropout=0.2)
        # self.conv2 = GATConv(256 * 8, self.hidden_size, heads=1, concat=False, dropout=0.2)

        # self.conv1 = GCNConv(self.hidden_size, 256)
        # self.conv2 = GCNConv(256, self.hidden_size)

        self.conv1 = CustomGCNConv(self.hidden_size, 256)
        self.conv2 = CustomGCNConv(256, self.hidden_size)

        self.convgcn1 = GCNConv(self.hidden_size, 256)
        self.convgcn2 = GCNConv(256, self.hidden_size)
        #新增
        self.embedding_cpu = nn.Embedding(num_cpu+1,emb_size)
        self.embedding_mem = nn.Embedding(num_mem+1,emb_size)
        self.embedding_gram = nn.Embedding(num_g+1,emb_size)
        # torch.nn.init.xavier_uniform_(self.embedding_gram.weight)
        self.p_layer = Linear(768, 256)
        self.q_layer = Linear(768, 256)



    def forward(self, p, q, g, c, m, r, q_next, p_next):  #p:学生作答的代码，q：题干（当知识点用） g：语法得分 c:cpu耗时，m：内存消耗  r：学生回答正确与否
        p_emb_bert = torch.stack([torch.stack([num2codeemb(int(num)) for num in row]) for row in p])  #使用codebert对问题进行编码
        # q_emb_bert = torch.stack([torch.stack([num2skillemb(int(num)) for num in row]) for row in q])     #使用codebert对知识点进行编码
        p_emb_bert = p_emb_bert.to(torch.float32)
        p_emb_bert=p_emb_bert.to(device)
        # q_emb_bert = q_emb_bert.to(torch.float32)
        # q_emb_bert = q_emb_bert.to(device)
        p_emb=self.p_layer(p_emb_bert)
        # q_emb = self.q_layer(q_emb_bert)


        # p_emb=self.embedding_problem(p)  #这个是直接对问题数字进行embedding 和上面使用codebert进行编码的方式 二选一
        q_emb = self.embedding_question(q)  #这个是直接对知识点数字进行embedding 和上面使用codebert进行编码的方式 二选一
        g_emb = self.embedding_gram(g)
        c_emb = self.embedding_cpu(c)
        m_emb =  self.embedding_mem(m)

        r_emb = self.answer_embedding(r)
        q_next_emb = self.embedding_question(q_next)
        p_next_emb = self.embedding_problem(p_next)

        #新增 逻辑/思路轨迹转换加权图GCN  起个名字叫：技能转换图
        batch_edge_index_p ,edge_weight= batch_genr_edge_index(p,q)
        every_stu_same_code__con_data = torch.cat((p_emb,g_emb), dim=-1)  # 和学生答题的code，语法得分，cpu，内存 ,c_emb,m_emb
        same_code_con_data_fc = self.same_code_concat_layer(every_stu_same_code__con_data)
        # 批量装载数据和对应的边
        batch_data_list = []
        batch_p_edge_weight = []
        for i in range(len(same_code_con_data_fc)):
            data_obj = Data(x=same_code_con_data_fc[i], edge_index=batch_edge_index_p[i])
            batch_data_list.append(data_obj)
            # batch_p_edge_weight.append(batch_edge_weight)
        batch_same_code = Batch.from_data_list(batch_data_list)
        # batch_same_weight = Batch.from_data_list(batch_p_edge_weight)
        batch_x, batch_edge_index = batch_same_code.x.to(device), batch_same_code.edge_index.to(device)
        # 使用GAT进行情感的聚合和更新
        x = F.dropout(batch_x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, batch_edge_index,edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, batch_edge_index,edge_weight)  # Output x is the new node features
        same_skill_state = x.view(-1, 219, 256)  #同一个学生练习序列内部     同一个知识点的能力表现

        #新增GCN  知识点之间的 相似性 没有谁比谁重要 所以 就不用GAT了  起个名字叫：技能相关图

        batch_similar_edge_index_p = batch_similar_genr_edge_index(q)
        every_stu_similar_code__con_data = torch.cat((q_emb,g_emb), dim=-1)  # 和学生答题的code，语法得分，cpu，内存
        similar_code_con_data_fc = self.similar_code_concat_layer(every_stu_similar_code__con_data)
        # 批量装载数据和对应的边
        batch_similar_data_list = []
        for i in range(len(similar_code_con_data_fc)):
            similar_data_obj = Data(x=similar_code_con_data_fc[i], edge_index=batch_similar_edge_index_p[i])
            batch_similar_data_list.append(similar_data_obj)
        batch_simalr_code= Batch.from_data_list(batch_similar_data_list)
        batch_similar_x, batch_similar_edge_index = batch_simalr_code.x.to(device), batch_simalr_code.edge_index.to(device)
        # 使用GCN进行情感的聚合和更新
        x = F.dropout(batch_similar_x, p=0.6, training=self.training)
        x = F.elu(self.convgcn1(x, batch_similar_edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.convgcn2(x, batch_similar_edge_index)  # Output x is the new node features
        similar_skill_state = x.view(-1, 219, 256)  # 同一个学生练习序列内部     同一个知识点的能力表现





        concatenated_qr = torch.cat((p_emb,q_emb,similar_skill_state,same_skill_state,r_emb), dim=-1)#,similar_skill_state, same_skill_state,
        # concatenated_qr = p_emb + q_emb + similar_skill_state + same_skill_state + r_emb

        #concatenated_pqr_dkt =torch.cat(( q_emb, r_emb), dim=-1)  #测试DKT的性能
        h, _ = self.lstm_layer(concatenated_qr)
        h = self.dropout_layer(h)
        h = torch.cat((h,q_next_emb), dim=-1)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        # Create model
        # d_model = concatenated_qr.size(-1)
        # nhead = 8
        # num_layers = 6
        # model = MyModel(d_model, nhead, num_layers).to(device)
        #
        # # Run model
        # y = model(p_emb.to(device), q_emb.to(device), similar_skill_state.to(device),same_skill_state.to(device), r_emb.to(device), q_next_emb.to(device)).to(device)

        return y


#新增 直接预测下一个知识点
def process_raw_pred_one(question, true_answer, answer):  #question, true_answer是一个学生所有的知识点和对应的答案，answer从第二个知识点开始的预测值
    mask = torch.zeros_like(question, dtype=torch.bool)
    mask[question != 0] = True  #找出一个学生所有知识点中非填充的知识点，为了下面的找真正知识点的真实值和预测值做准备
    count = torch.sum(mask)     #统计一个学生所有知识点中非填充的知识点的个数
    final_true_answer = torch.masked_select(true_answer[1:count], mask[1:count]).to(device) #[1:count]从第二个知识点对应的答案开始找
    final_answer = torch.masked_select(torch.flatten(answer)[0:count-1], mask[0:count-1]).to(device)#[0:count-1] 从第一个预测值开始使用，count-1是因为本来answer就是少一位 你可以用count=seqlen来举例，马上理解
    return final_answer, final_true_answer


class myKT_DKT(KTM):
    def __init__(self, num_problem,num_questions,num_g,num_cpu,num_mem,emb_size):
        super(myKT_DKT, self).__init__()
        self.num_questions = num_questions
        self.dkt_model = DKT(num_problem,num_questions,num_g,num_cpu,num_mem, emb_size).to(device)

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.003) -> ...:
        auc_max=0
        count_e=0
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)

        for e in range(epoch):
            all_pred, all_target = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                all_pred = torch.Tensor([]).to(device)  # 清空all_pred张量
                all_target = torch.Tensor([]).to(device)  # 清空all_target张量
                true_length,batch_p, batch_q, batch_g,batch_c,batch_m,batch_a = batch
                # 将每个字符串转换为整数列表

                batch_p = [list(map(int, p.split(','))) for p in batch_p]
                batch_q = [list(map(int, kp.split(','))) for kp in batch_q]
                batch_g = [list(map(int, g.split(','))) for g in batch_g]
                batch_c = [list(map(int, c.split(','))) for c in batch_c]
                batch_m = [list(map(int, m.split(','))) for m in batch_m]
                batch_a = [list(map(int, answer.split(','))) for answer in batch_a]


                # 将列表转换为张量（tensor）
                batch_p = torch.tensor(batch_p).to(device)
                batch_q = torch.tensor(batch_q).to(device)
                batch_g = torch.tensor(batch_g).to(device)
                batch_c = torch.tensor(batch_c).to(device)
                batch_m = torch.tensor(batch_m).to(device)
                batch_a = torch.tensor(batch_a).to(device)
                #新增
                batch_q_next = batch_q[:,1:batch_q.shape[1]]
                batch_p_next = batch_p[:, 1:batch_p.shape[1]]




                pred_y = self.dkt_model(batch_p[:,0:batch_p.shape[1]-1].to(device),batch_q[:,0:batch_q.shape[1]-1].to(device),batch_g[:,0:batch_g.shape[1]-1].to(device), batch_c[:,0:batch_c.shape[1]-1].to(device),batch_m[:,0:batch_m.shape[1]-1].to(device),batch_a[:,0:batch_q.shape[1]-1].to(device), batch_q_next.to(device),batch_p_next.to(device))

                batch_size = batch_q.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred_one(batch_q[student].to(device), batch_a[student].to(device),
                                                   pred_y[student].to(device))
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float().to(device)])
                # print(f"预测长度{all_pred.size()}")
                loss = loss_function(all_pred, all_target)
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                # torch.cuda.empty_cache()
                optimizer.step()

            # 再次检查特定层的参数是否发生了更新
            # for name, param in self.dkt_model.named_parameters():
            #     if name == 'out_cluser.weight':
            #         if param.grad is not None:
            #             print("fc1.weight 已更新")
            #         else:
            #             print("fc1.weight 未更新")
            #     elif name == 'out_layer.weight':
            #         if param.grad is not None:
            #             print("fc2.weight 已更新")
            #         else:
            #             print("fc2.weight 未更新")

            print("[Epoch %d] LogisticLoss: %.6f" % (e, loss))

            if test_data is not None:
                auc,acc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f , acc: %.6f " % (e, auc,acc))
                if auc > auc_max:
                    auc_max = auc
                    count_e = e+1
        print(f"最大的auc是在第{count_e}轮出现的：{auc_max}")

    def eval(self, test_data) -> float:
        # self.dkt_model.eval()

        y_pred = torch.Tensor([]).to(device)
        y_truth = torch.Tensor([]).to(device)
        for batch in tqdm.tqdm(test_data):
            true_length, batch_p, batch_q, batch_g, batch_c, batch_m, batch_a = batch
            # 将每个字符串转换为整数列表

            batch_p = [list(map(int, p.split(','))) for p in batch_p]
            batch_q = [list(map(int, kp.split(','))) for kp in batch_q]
            batch_g = [list(map(int, g.split(','))) for g in batch_g]
            batch_c = [list(map(int, c.split(','))) for c in batch_c]
            batch_m = [list(map(int, m.split(','))) for m in batch_m]
            batch_a = [list(map(int, answer.split(','))) for answer in batch_a]

            # 将列表转换为张量（tensor）
            batch_p = torch.tensor(batch_p).to(device)
            batch_q = torch.tensor(batch_q).to(device)
            batch_g = torch.tensor(batch_g).to(device)
            batch_c = torch.tensor(batch_c).to(device)
            batch_m = torch.tensor(batch_m).to(device)
            batch_a = torch.tensor(batch_a).to(device)
            # 新增
            batch_q_next = batch_q[:, 1:batch_q.shape[1]]
            batch_p_next = batch_p[:, 1:batch_p.shape[1]]

            pred_y = self.dkt_model(batch_p[:, 0:batch_p.shape[1] - 1].to(device),
                                    batch_q[:, 0:batch_q.shape[1] - 1].to(device),
                                    batch_g[:, 0:batch_g.shape[1] - 1].to(device),
                                    batch_c[:, 0:batch_c.shape[1] - 1].to(device),
                                    batch_m[:, 0:batch_m.shape[1] - 1].to(device),
                                    batch_a[:, 0:batch_q.shape[1] - 1].to(device), batch_q_next.to(device),
                                    batch_p_next.to(device))

            batch_size = batch_q.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred_one(batch_q[student].to(device), batch_a[student].to(device),
                                                   pred_y[student].to(device))
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])

        accuracy = accuracy_score(y_truth.cpu().detach().numpy(), y_pred.cpu().detach().numpy().round())

        return roc_auc_score(y_truth.cpu().detach().numpy(), y_pred.cpu().detach().numpy()),accuracy

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

if __name__ == "__main__":
    pass




