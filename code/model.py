from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from load_positive_negative_sample import *
from scipy.sparse import *
import numpy as np

# positive_sample torch.Size([3])          1024*3
# negative_sample_head torch.Size([128])   1024*128
# negative_sample_tail torch.Size([128])   1024*128

# tail-match
# head_tensor_positive 1024*1*500
# relation_tensor_positive 1024*1*500
# tail_tensor_positive 1024*1*500
# tail_tensor_negative 1024*128*500

# head-match
# head_tensor_positive 1024*1*500
# relation_tensor_positive 1024*1*500
# tail_tensor_positive 1024*1*500
# head_tensor_negative 1024*128*500


class Count_Score(nn.Module):
    def __init__(self,gamma,whcih_score):
        super(Count_Score, self).__init__()
        self.gamma = gamma
        self.which_score = whcih_score

    def forward(self, head_tensor, relation_tensor, tail_tensor, mode):

        if mode == "head-match":  # head_tensor: 1024*128*500;  relation_tensor: 1024*1*500;  tail_tensor: 1024*1*500
            score = head_tensor + (relation_tensor - tail_tensor)
        else:
            score = (head_tensor + relation_tensor) - tail_tensor  # 1024*128*500/1024*1*500

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)  # 2014*128/1024*1
        return score

class Sample_Score(nn.Module):
    def __init__(self, G2_adj, G1_sub1_adj, G1_graph_sub2, sample_num, num_node_G2, G2_links,G1_sub1_links,gamma, num_relation_G2, num_relation_G1_sub1, num_node_G1_sub1, num_type, negative_sample_number):
        super(Sample_Score, self).__init__()
        self.G2_adj = G2_adj
        self.G1_sub1_adj = G1_sub1_adj
        self.G1_graph_sub2 = G1_graph_sub2
        self.sample_num = sample_num
        self.num_node_G2 = num_node_G2
        self.temp_nonzero_len_part1 = len(G2_adj.nonzero()[0])
        self.temp_nonzero_len_part2 = len(G1_sub1_adj.nonzero()[0])
        self.G2_links = G2_links
        self.G1_sub1_links = G1_sub1_links
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.num_relation_G2 = num_relation_G2
        self.num_node_G1_sub1 = num_node_G1_sub1
        self.num_type = num_type
        self.negative_sample_number = negative_sample_number
        row_l_tmp = self.G2_adj.nonzero()[0]
        col_l_tmp = self.G2_adj.nonzero()[1]
        G2_adj_re = np.ones((self.num_node_G2, self.num_node_G2))
        for i in range(len(row_l_tmp)):
            G2_adj_re[row_l_tmp[i], col_l_tmp[i]] = G2_adj_re[row_l_tmp[i], col_l_tmp[i]] - 1
        self.G2_adj_re = G2_adj_re
        self.temp_nonzero_len_part1_re = len(G2_adj_re.nonzero()[0])
        self.num_relation_G1_sub1 = num_relation_G1_sub1
        row_l_tmp = self.G1_sub1_adj.nonzero()[0]
        col_l_tmp = self.G1_sub1_adj.nonzero()[1]
        G1_sub1_adj_re = np.ones((self.num_node_G1_sub1, self.num_node_G1_sub1))
        for i in range(len(row_l_tmp)):
            G1_sub1_adj_re[row_l_tmp[i], col_l_tmp[i]] = G1_sub1_adj_re[row_l_tmp[i], col_l_tmp[i]] - 1
        self.G1_sub1_adj_re = G1_sub1_adj_re
        self.temp_nonzero_len_part2_re = len(G1_sub1_adj_re.nonzero()[0])

    def sample_tensor(self,all_node_embedding, relation_embedding, start_idx, end_idx, sample_num, adj, links, embedding_dim, negative_sample_number, true_tail, true_head):
        # sample from G2_adj # 26078*26078 # count score of tuple: entity-relation-entity
        sample_list_entity_couple = []
        sample_list_id = []
        sample_index = np.random.sample(range(start_idx, end_idx), sample_num)
        for i in sample_index:
            sample_list_entity_couple.append((adj.nonzero()[0][i], adj.nonzero()[1][i]))
            sample_list_id.append(links[i])

        head_ndarray = np.zeros((sample_num, embedding_dim))  # 1024*500
        relation_ndarray = np.zeros((sample_num, embedding_dim))  # 1024*500
        tail_ndarray = np.zeros((sample_num, embedding_dim))  # 1024*500
        tail_ndarray_negative = np.zeros((sample_num, negative_sample_number, embedding_dim))  # 1024*128*500
        head_ndarray_negative = np.zeros((sample_num, negative_sample_number, embedding_dim))  # 1024*128*500

        for i in range(self.sample_num):
            head_ndarray[i, :] = all_node_embedding[sample_list_entity_couple[i][0]]
            relation_ndarray[i, :] = relation_embedding[sample_list_id[i]]
            tail_ndarray[i, :] = all_node_embedding[sample_list_entity_couple[i][1]]
            tail_ndarray_negative[i, :, :] = negative_sample_tail_fun(negative_sample_number, start_idx, end_idx, true_tail, sample_list_entity_couple, sample_list_id, i, all_node_embedding, embedding_dim)
            head_ndarray_negative[i, :, :] = negative_sample_head_fun(negative_sample_number, start_idx, end_idx, true_head, sample_list_entity_couple, sample_list_id, i, all_node_embedding, embedding_dim)

        head_tensor_positive = torch.from_numpy(head_ndarray).unsqueeze(1)  # 1024*1*500
        relation_tensor_positive = torch.from_numpy(relation_ndarray).unsqueeze(1)  # 1024*1*500
        tail_tensor_positive = torch.from_numpy(tail_ndarray).unsqueeze(1)  # 1024*1*500
        tail_tensor_negative = torch.from_numpy(tail_ndarray_negative)  # 1024*128*500
        head_tensor_negative = torch.from_numpy(head_ndarray_negative)  # 1024*128*500
        return head_tensor_positive, relation_tensor_positive, tail_tensor_positive, tail_tensor_negative, head_tensor_negative


    def forward(self, all_node_embedding, relation_embedding_G2, relation_embedding_G1_type_of, relation_embedding_G1_ty_ty, true_tail_en_en, true_head_en_en, true_tail_ty_ty, true_head_ty_ty, list_type_G1_sub2_and_sub3, true_tail_en_ty, list_entity_G1_sub2_and_sub3, true_head_en_ty):
        count_score = Count_Score(self.gamma, 1)  # part=1, gamma=12 tensor, score1
        sample_tensor = self.sample_tensor

        # sample from G2_adj # 26078*26078 # count score of tuple: entity-relation-entity

        head_tensor_positive, relation_tensor_positive, tail_tensor_positive, tail_tensor_negative, head_tensor_negative = sample_tensor(all_node_embedding, relation_embedding_G2, 0, self.num_node_G2-1, self.sample_num, self.G2_adj, self.G2_links, self.embedding_dim, self.negative_sample_number, true_tail_en_en, true_head_en_en)
        positive_score_1 = count_score(self, head_tensor_positive, relation_tensor_positive, tail_tensor_positive, 'single')  # positive # 1024*1
        negative_score_1_tail = count_score(self, head_tensor_positive, relation_tensor_positive, tail_tensor_negative, 'tail-match')  # tail-match # 1024*128
        negative_score_1_head = count_score(self, head_tensor_positive, relation_tensor_positive, head_tensor_negative, 'head-match')  # head-match # 1024*128

        # sample from G1_sub1_adj # 906*906 # count score of tuple: type-relation-type
        head_tensor_positive, relation_tensor_positive, tail_tensor_positive, tail_tensor_negative, head_tensor_negative = sample_tensor(all_node_embedding, relation_embedding_G1_ty_ty, self.num_node_G2, int(self.num_node_G2 + self.num_type-1), self.sample_num, self.G1_sub1_adj, self.G1_sub1_links, self.embedding_dim, self.negative_sample_number, true_tail_ty_ty, true_head_ty_ty)
        positive_score_2 = count_score(self, head_tensor_positive, relation_tensor_positive, tail_tensor_positive, 'single')  # positive # 1024*1
        negative_score_2_tail = count_score(self, head_tensor_positive, relation_tensor_positive, tail_tensor_negative, 'tail-match')  # tail-match  # 1024*128
        negative_score_2_head = count_score(self, head_tensor_positive, relation_tensor_positive, head_tensor_negative, 'head-match')  # head-match  # 1024*128

        # sample from G1_graph_sub2 # 8948*106
        # positive
        keyli = self.G1_graph_sub2.keys()
        len = len(keyli)
        sample_index = random.sample(range(0, len - 1), self.sample_num)
        head_ndarray = np.zeros((self.sample_num, self.embedding_dim))  # 1024*500
        relation_ndarray = np.zeros((self.sample_num, self.embedding_dim))  # 1024*500
        tail_ndarray = np.zeros((self.sample_num, self.embedding_dim))  # 1024*500
        tail_ndarray_negative = np.zeros((self.sample_num, self.negative_sample_number, self.embedding_dim))  # 1024*(128/4)*500
        head_ndarray_negative = np.zeros((self.sample_num, self.negative_sample_number, self.embedding_dim))  # 1024*(128/4)*500
        for i in sample_index:
            head_ndarray[i,:] = all_node_embedding[keyli[i]]
            relation_ndarray[i,:] = relation_embedding_G1_type_of
            tail_ndarray[i,:] = all_node_embedding[self.G1_graph_sub2[keyli[i]][0]]

            tail_ndarray_negative[i,:,:] = negative_sample_tail_fun_en_ty(self.negative_sample_number, self.embedding_dim, list_type_G1_sub2_and_sub3, true_tail_en_ty, keyli[i], all_node_embedding)

            head_ndarray_negative[i,:,:] = negative_sample_head_fun_en_ty(self.negative_sample_number, self.embedding_dim, list_entity_G1_sub2_and_sub3, true_head_en_ty, self.G1_graph_sub2[keyli[i]][0], all_node_embedding)
        head_tensor_positive = torch.from_numpy(head_ndarray).unsqueeze(1)  # 1024*1*500
        relation_tensor_positive = torch.from_numpy(relation_ndarray).unsqueeze(1)  # 1024*1*500
        tail_tensor_positive = torch.from_numpy(tail_ndarray).unsqueeze(1)  # 1024*1*500
        tail_tensor_negative = torch.from_numpy(tail_ndarray_negative)  # 1024*(32)*500
        head_tensor_negative = torch.from_numpy(head_ndarray_negative)  # 1024*(32)*500
        positive_score_3 = count_score(self, head_tensor_positive, relation_tensor_positive, tail_tensor_positive, 'single')  # positive # 1024*1
        negative_score_3_tail = count_score(self, head_tensor_positive, relation_tensor_positive, tail_tensor_negative, 'tail-match')  # tail-match  # 1024*32
        negative_score_3_head = count_score(self, head_tensor_positive, relation_tensor_positive, head_tensor_negative, 'head-match')  # head-match  # 1024*32
        return positive_score_1, negative_score_1_tail, negative_score_1_head, positive_score_2, negative_score_2_tail, negative_score_2_head, positive_score_3, negative_score_3_tail, negative_score_3_head


class Layer(nn.Module):
    def __init__(self, G2_adj, G2_three_dim_node_weights, G2_three_dim_relation, embedding_dim, number_node_G2,G1_sub1_adj, num_type_node_G1, G1_graph_sub2, G1_graph_sub3, num_entity_node_G1, num_type_node_G1_sub2_and_sub3):
        super(Layer, self).__init__()
        self.G2_adj = G2_adj
        self.G2_three_dim_relation = G2_three_dim_relation

        self.embedding_dim = embedding_dim
        self.number_node_G2 = number_node_G2
        self.G1_sub1_adj = G1_sub1_adj
        self.num_type_node_G1 = num_type_node_G1 # 911
        self.start_idx = number_node_G2
        self.G1_graph_sub2 = G1_graph_sub2
        self.G1_graph_sub2 = G1_graph_sub2
        self.G1_graph_sub3 = G1_graph_sub3
        self.num_entity_node_G1 = num_entity_node_G1
        self.num_type_node_G1_sub2_and_sub3 = num_type_node_G1_sub2_and_sub3
        self.G2_three_dim_node_weights = G2_three_dim_node_weights
        self.g2 = G2_update(self.G2_adj, self.G2_three_dim_node_weights, self.G2_three_dim_relation, self.embedding_dim, self.number_node_G2)
        self.g1_sub1 = G1_sub1_update(self.G1_sub1_adj, self.embedding_dim, self.num_type_node_G1, self.number_node_G2)
        self.g1_sub2_and_sub3 = G1_sub2_and_sub3_update(self.G1_graph_sub2, self.G1_graph_sub3, self.embedding_dim, self.num_entity_node_G1, self.num_type_node_G1_sub2_and_sub3)
    def forward(self, all_node_embedding):
        #  G2 更新 all_node_embedding中的前26078行（G2中的所有entity）
        all_node_embedding = self.g2(all_node_embedding)
        #  G1_sub1  更新all_node_embedding中的后906行  （G1中的所有的type）
        all_node_embedding = self.g1_sub1(all_node_embedding)
        #  G1_sub2_and_sub3  更新all_node_embedding中的（26078：26078+106）行type 和 前26078行中的部分8948行 entity
        all_node_embedding = self.g1_sub2_and_sub3(all_node_embedding)
        return all_node_embedding


class G2_update(nn.Module):
    def __init__(self, G2_adj, G2_three_dim_node_weights, G2_three_dim_relation, embedding_dim, number_node_G2):
        super(G2_update, self).__init__()
        self.G2_adj = G2_adj
        self.number_node_G2 = number_node_G2
        self.G2_three_dim_relation = G2_three_dim_relation
        self.embedding_dim = embedding_dim
        self.G2_three_dim_node_weights = G2_three_dim_node_weights

    def forward(self, all_node_embedding):
        #  三维的矩阵，第三纬度是实体的embedding, 26078*26078*500
        G2_three_dim_node = torch.from_numpy(np.zeros((self.number_node_G2, self.number_node_G2, self.embedding_dim)))

        for i in range(self.number_node_G2):
            for j in range(self.number_node_G2):
                G2_three_dim_node[j, i, :] = all_node_embedding[i]
        #  二纬矩阵，26078*500，node embedding
        node_embedding_G2 = all_node_embedding[0:self.number_node_G2]
        #  26078*500
        new_node_embedding_G2 = (torch.mul(G2_three_dim_node,self.G2_three_dim_node_weights) + self.G2_three_dim_relation).sum(1) + node_embedding_G2
        #  更新all_node_embedding中的前26078行
        all_node_embedding[0:self.number_node_G2] = new_node_embedding_G2
        return all_node_embedding

class G1_sub1_update(nn.Module):
    def __init__(self, G1_sub1_adj, embedding_dim, num_type_node_G1, start_idx):
        super(G1_sub1_update, self).__init__()
        self.G1_sub1_adj = G1_sub1_adj
        self.number_node_G1_sub1 = num_type_node_G1
        self.embedding_dim = embedding_dim
        self.start_idx =start_idx

    def forward(self, all_node_embedding):
        #  三纬的矩阵， 第三纬度是type的embedding, 911*911*500
        G1_sub1_three_dim_node = np.zeros((self.number_node_G1_sub1, self.number_node_G1_sub1, self.embedding_dim))
        for i in range(self.number_node_G1_sub1):
            G1_sub1_three_dim_node[:,i,:] = all_node_embedding[self.start_idx + i]
        #  911*500
        node_embedding_G1_sub1 = all_node_embedding[self.start_idx : len(self.all_node_embedding)]
        #  specific--->common
        #  911*500
        new_node_embedding_G1_sub1 = node_embedding_G1_sub1 + (G1_sub1_three_dim_node).sum(1)
        #  更新all_node_embedding中的后911行
        all_node_embedding[self.start_idx : len(all_node_embedding)] = new_node_embedding_G1_sub1
        #  common---->specific
        nonzero_row = self.G1_sub1_adj.nonzero()[0]
        sum_arr = np.ones(self.number_node_G1_sub1, 1) # 911*1
        sum_dict = dict()
        for item in nonzero_row:
            if item not in sum_dict.keys():
                sum_dict[item] = 0
            sum_dict[item] = sum_dict[item] + 1
        for key, value in sum_dict.items():
            sum_arr[key,0] = sum_arr[key,0] + value
        # 911*200
        Iarr = np.ones((self.number_node_G1_sub1, self.embedding_dim))
        new_node_embedding_G1_sub1 = torch.mul(new_node_embedding_G1_sub1, Iarr - ((G1_sub1_three_dim_node).sum(1) / sum_arr))
        #  更新all_node_embedding中的后911行
        all_node_embedding[self.start_idx : len(all_node_embedding)] = new_node_embedding_G1_sub1
        return all_node_embedding

class G1_sub2_and_sub3_update(nn.Module):
    def __init__(self, G1_graph_sub2, G1_graph_sub3, embedding_dim, num_entity_node_G1, num_type_node_G1_sub2_and_sub3):
        super(G1_sub2_and_sub3_update, self).__init__()
        self.G1_graph_sub2 = G1_graph_sub2
        self.G1_graph_sub3 = G1_graph_sub3
        self.embedding_dim = embedding_dim
        self.num_entity_node_G1 = num_entity_node_G1
        self.num_type_node_G1_sub2_and_sub3 = num_type_node_G1_sub2_and_sub3
    def forward(self,all_node_embedding):
        left_specific = list()  # 8498
        right_common = list()  # 106

        # specific--->common
        # 8948 * 106
        arr_G1_graph_sub2 = np.zeros((self.num_entity_node_G1, self.num_type_node_G1_sub2_and_sub3))

        for left_spec, right_comm in self.G1_graph_sub2.items():
            if left_spec not in left_specific:
                left_specific.append(left_spec)
            for values in right_comm:
                if values not in right_common:
                    right_common.append(values)

                arr_G1_graph_sub2[left_specific.index(left_spec), right_common.index(values)] = 1
        # 三维 8498*106*500 第三纬是entity（specific）的embedding
        three_dim_specific_G1_sub2 = np.ones((self.num_entity_node_G1, self.num_type_node_G1_sub2_and_sub3, self.embedding_dim))

        rowarr = arr_G1_graph_sub2.nonzero()[0]
        colarr = arr_G1_graph_sub2.nonzero()[1]
        for i in range(len(rowarr)):
            three_dim_specific_G1_sub2[rowarr[i], colarr[i], :] = all_node_embedding[left_specific[rowarr[i]], :]
        #  106*500
        common_embedding_G1_sub2 = all_node_embedding[right_common]
        #  106*500
        new_common_embedding_G1_sub2 = common_embedding_G1_sub2 + (three_dim_specific_G1_sub2.sum(0))
        #  更新106 type 按照right_common list里的顺序
        all_node_embedding[right_common] = new_common_embedding_G1_sub2

        # common--->specific
        # 106 * 8498
        arr_G1_graph_sub3 = np.zeros((self.num_type_node_G1_sub2_and_sub3, self.num_entity_node_G1))
        left_common = list()  # 106
        right_specific = list()  # 8948
        for left_comm, right_spec in self.G1_graph_sub3.items():
            if left_comm not in left_common:
                left_common.append(left_comm)
            for values in right_specific:
                if values not in right_specific:
                    right_specific.append(values)

                arr_G1_graph_sub3[left_common.index(left_comm), right_specific.index(values)] = 1
        # 三维 106*8948*500 第三纬是type（common）的embedding
        three_dim_common_G1_sub3 = np.ones(
            (self.num_type_node_G1_sub2_and_sub3, self.num_entity_node_G1, self.embedding_dim))

        rowarr = arr_G1_graph_sub3.nonzero()[0]
        colarr = arr_G1_graph_sub3.nonzero()[1]
        for i in range(len(rowarr)):
            three_dim_common_G1_sub3[rowarr[i], colarr[i], :] = all_node_embedding[left_common[rowarr[i]], :]

        sum_arr = np.ones(self.num_entity_node_G1, 1)  # 8948*1
        sum_dict = dict()
        for item in colarr:
            if item not in sum_dict.keys():
                sum_dict[item] = 0
            sum_dict[item] = sum_dict[item] + 1
        for key, value in sum_dict.items():
            sum_arr[key, 0] = sum_arr[key, 0] + value
        # 8948*500
        Iarr = np.ones((self.num_entity_node_G1, self.embedding_dim))
        #  8948*500
        specific_embedding_G1_sub3 = all_node_embedding[right_specific]
        #  8948*500
        new_specific_embedding_G1_sub3 = torch.mul(specific_embedding_G1_sub3, Iarr - ((three_dim_common_G1_sub3).sum(0) / sum_arr))
        #  更新8948 entity 按照right_specific list里的顺序
        all_node_embedding[right_specific] = new_specific_embedding_G1_sub3
        return all_node_embedding

class GCN_DD(nn.Module):

    def __init__(self, G2_adj, embedding_dim, number_node_G2,G1_sub1_adj, num_type_node_G1, G1_graph_sub2, G1_graph_sub3, num_entity_node_G1, num_type_node_G1_sub2_and_sub3, sample_num, num_node_G2, G2_links,G1_sub1_links,gamma, num_relation_G2, num_relation_G1_sub1, num_node_G1_sub1, num_type,negative_sample_number,true_tail_en_en, true_head_en_en, true_tail_ty_ty, true_head_ty_ty, list_type_G1_sub2_and_sub3, true_tail_en_ty, list_entity_G1_sub2_and_sub3, true_head_en_ty):
        super(GCN_DD, self).__init__()
        node_embedding_G2 = np.load('/Users/bubuying/PycharmProjects/DG/data/entity_embedding/node_embedding.npy')
        node_embedding_type = np.load('/Users/bubuying/PycharmProjects/DG/data/type_embedding/node_embedding.npy')

        self.all_node_embedding = nn.Parameter(torch.from_numpy(np.vstack((node_embedding_G2, node_embedding_type))))

        self.relation_embedding_G2 = nn.Parameter(torch.from_numpy(np.load('/Users/bubuying/PycharmProjects/DG/data/entity_embedding/node_re_embedding.npy')))
        self.relation_embedding_G1_ty_ty = nn.Parameter(torch.from_numpy(np.load('/Users/bubuying/PycharmProjects/DG/data/type_embedding/ty_ty_relation_embedding.npy')),requires_grad=False)
        self.relation_embedding_G1_type_of = nn.Parameter(torch.zeros(1, embedding_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding_G1_type_of,
            a=-1,
            b=1
        )
        #  三维的矩阵，第三纬度是两个实体之间关系的embedding, 26078*26078*500
        G2_three_dim_relation = torch.from_numpy(np.zeros((G2_adj.shape[0], G2_adj.shape[0], embedding_dim), dtype=np.float32))
        G2_three_dim_node_weights = torch.from_numpy(np.zeros((G2_adj.shape[0], G2_adj.shape[0], embedding_dim), dtype=np.float32))
        row_array = G2_adj.nonzero()[0]
        col_array = G2_adj.nonzero()[1]
        for key, value in G2_links.items():
            x = int(key[0])
            y = int(key[1])
            # cur_embedding = torch.from_numpy(np.zeros((1,embedding_dim), dtype=np.float))
            for values in value:
                # import pdb
                # pdb.set_trace()
                G2_three_dim_relation[x,y,:] = G2_three_dim_relation[x,y,:] + self.relation_embedding_G2[int(values)]

            cur_weight = len(value)

            G2_three_dim_node_weights[x,y,:] = torch.from_numpy(np.array([cur_weight for x in range(500)]))



        self.G2_three_dim_relation = G2_three_dim_relation
        self.layer = Layer(G2_adj, G2_three_dim_node_weights, G2_three_dim_relation, embedding_dim, number_node_G2,G1_sub1_adj, num_type_node_G1, G1_graph_sub2, G1_graph_sub3, num_entity_node_G1, num_type_node_G1_sub2_and_sub3)
        self.sample_score = Sample_Score(G2_adj, G1_sub1_adj, G1_graph_sub2, sample_num, num_node_G2, G2_links,G1_sub1_links,gamma, num_relation_G2, num_relation_G1_sub1, num_node_G1_sub1, num_type, negative_sample_number)
        self.true_tail_en_en = true_tail_en_en
        self.true_head_en_en = true_head_en_en
        self.true_tail_ty_ty = true_tail_ty_ty
        self.true_head_ty_ty = true_head_ty_ty
        self.list_type_G1_sub2_and_sub3 = list_type_G1_sub2_and_sub3
        self.true_tail_en_ty = true_tail_en_ty
        self.list_entity_G1_sub2_and_sub3 = list_entity_G1_sub2_and_sub3
        self.true_head_en_ty = true_head_en_ty

    def forward(self):# ,all_node_embedding, relation_embedding_G2, relation_embedding_G1_type_of, relation_embedding_G1_ty_ty):
        out = self.layer(self.all_node_embedding)
        out = self.sample_score(out, self.relation_embedding_G2, self.relation_embedding_G1_type_of, self.relation_embedding_G1_ty_ty, self.true_tail_en_en, self.true_head_en_en, self.true_tail_ty_ty, self.true_head_ty_ty, self.list_type_G1_sub2_and_sub3, self.true_tail_en_ty, self.list_entity_G1_sub2_and_sub3, self.true_head_en_ty)
        return out