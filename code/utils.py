# coding:utf-8
import numpy as np
from numpy import random
from collections import OrderedDict
import networkx as nx
import logging
import os

def load_data(args, all_node2id, node2id_G2, type_node2id_G1, entity_node2id_G1, relation2id_G1_sub1, relation2id_G2):
    # get reverse
    node2id_G2_re = {v: k for k, v in node2id_G2.items()}
    type_node2id_G1_re = {v: k for k, v in type_node2id_G1.items()}
    relation2id_G1_sub1_re = {v: k for k, v in relation2id_G1_sub1.items()}
    relation2id_G2_re = {v: k for k, v in relation2id_G2.items()}

    num_all_node = len(all_node2id)
    num_entity_G2 = len(node2id_G2)
    num_relation_G2 = len(relation2id_G2)
    num_entity_G1 = len(entity_node2id_G1)
    num_type_G1 = len(type_node2id_G1)

    logging.info('Data Path: %s' % args.data_path)
    logging.info('#all nodes number: %d' % num_all_node)
    logging.info('#entity nodes number in G2: %d' %num_entity_G2)
    logging.info('#relations number in G2: %d' % num_relation_G2)
    logging.info('#entity nodes number in G1: %d' % num_entity_G1)
    logging.info('#type nodes number in G1: %d' % num_type_G1)

    # G2(entity-entity)
    G2_graph = OrderedDict()
    G2_links = OrderedDict()
    all_graph = OrderedDict()

    with open(os.path.join(args.data_path, 'train_entity_Graph.txt')) as fin:  # 332127
        for line in fin:
            en1, r, en2 = line.strip().split('\t')
            en1id = node2id_G2_re[en1]
            rid = relation2id_G2_re[r]
            en2id = node2id_G2_re[en2]
            if en1id not in G2_graph.keys():
                G2_graph[en1id] = set()
            G2_graph[en1id].add(en2id)

            if (en1id,en2id) not in G2_links:
                G2_links[(en1id,en2id)] = list()  # relation: 332127 # 位置是1的有 228619
            G2_links[(en1id,en2id)].append(rid)

            # all_graph
            if en1id not in all_graph.keys():
                all_graph[en1id] = list()
            all_graph[en1id].append(en2id)

    count = 0
    for key, values in G2_graph.items():
        count = count + len(values)
    print(count)

    G = nx.Graph()
    #  二维的邻接矩阵，稀疏矩阵csr 26078*26078, 1的数目是228619
    G2_adj = nx.adjacency_matrix(nx.from_dict_of_lists(G2_graph, nx.DiGraph()))


    # G1
    # (type-type)
    count_links_ty_ty = 0
    G1_graph_sub1 = OrderedDict()
    G1_sub1_links= []
    with open(os.path.join(args.data_path_bef, 'yago_ontonet.txt')) as fin: # 8902
        for line in fin:
            ty1, r, ty2 = line.strip().split('\t')
            ty1id = type_node2id_G1_re[ty1]
            rid = relation2id_G1_sub1_re[r]
            ty2id = type_node2id_G1_re[ty2]
            if ty1id not in G1_graph_sub1.keys():
                G1_graph_sub1[ty1id] = list()
            G1_graph_sub1[ty1id].append(ty2id)
            G1_sub1_links.append(rid)
            count_links_ty_ty = count_links_ty_ty + 1
            # all_graph
            if ty1id not in all_graph.keys():
                all_graph[ty1id] = list()
            all_graph[ty1id].append(ty2id)

    # csr: 906*906
    G1_sub1_adj = nx.adjacency_matrix(nx.from_dict_of_lists(G1_graph_sub1))

    # (entity-type)
    count_links_en_ty_2 = 0
    count_links_en_ty_3 = 0

    G1_graph_sub2 = OrderedDict()
    G1_graph_sub3 = OrderedDict()
    with open(os.path.join(args.data_path_bef, 'yago_InsType_mini.txt')) as fin: # 9962
        for line in fin:
            en, _, ty = line.strip().split('\t')
            enid = node2id_G2_re[en]
            tyid = type_node2id_G1_re[ty]
            if enid not in G1_graph_sub2.keys():
                G1_graph_sub2[enid] = list()
            G1_graph_sub2[enid].append(tyid)
            count_links_en_ty_2 = count_links_en_ty_2 + 1
            # all_graph
            if enid not in all_graph.keys():
                all_graph[enid] = list()
            all_graph[enid].append(tyid)

            if tyid not in G1_graph_sub3.keys():
                G1_graph_sub3[tyid] = list()
            G1_graph_sub3[tyid].append(enid)
            count_links_en_ty_3 = count_links_en_ty_3 + 1
            # all_graph
            if tyid not in all_graph.keys():
                all_graph[tyid] = list()
            all_graph[tyid].append(enid)

    # all_graph = G2_graph + G1_graph_sub1 + G1_graph_sub2 + G1_graph_sub3

    # csr: (26078+911)*(26078+911)  26989
    all_adj = nx.adjacency_matrix(nx.from_dict_of_lists(all_graph))




    # all_adj: 26989*26989  (26078+911)
    # G2_adj: 26078*26078
    # G2_three_dim: 26078*26078*500
    # G1_sub1_adj:  906*906
    # type-type里面有5个游离的 在type图里没有link, 但是在entity-type里面有link。
    # 所有G1_sub1_adj是906*906，但是all_adj是（26078+911）是911不是906

    # true_tail_en_en 是dict
    # true_head_en_en 是dict
    # true_tail_ty_ty 是dict
    # true_head_ty_ty 是dict
    # true_tail_en_ty 是dict
    # true_head_en_ty 是dict
    # list_type_G1_sub2_and_sub3 是list
    # list_entity_G1_sub2_and_sub3 是list
    all_en_dict = OrderedDict() # all_entity
    with open('/Users/bubuying/PycharmProjects/DG/data/yago_result/final_entity_order.txt') as fin:
        for line in fin:
            id, en = line.strip().split('\t')
            all_en_dict[int(id)] = en
            len_1 = len(all_en_dict)

    all_en_re = OrderedDict() # all_entity_relation
    with open('/Users/bubuying/PycharmProjects/DG/data/yago_result/ffinal_en_relation_order.txt') as fin:
        for line in fin:
            id, r = line.strip().split('\t')
            all_en_re[int(id)] = r
            len_2 = len(all_en_re)

    all_ty_dict = OrderedDict()
    base_id = len(all_en_dict) # all_type
    with open('/Users/bubuying/PycharmProjects/DG/data/yago_result/final_type_order.txt') as fin:
        for line in fin:
            id, ty = line.strip().split('\t')
            all_ty_dict[int(id)+int(base_id)] = ty
            len_3 = len(all_ty_dict)

    all_ty_re = OrderedDict() # all_type_relation
    with open('/Users/bubuying/PycharmProjects/DG/data/yago_result/ffinal_ty_relation_order.txt') as fin:
        for line in fin:
            id, r = line.strip().split('\t')
            all_ty_re[int(id)] = r
            len_4 = len(all_ty_re)

    # reverse
    all_en_dict_rev = {v: k for k, v in all_en_dict.items()}
    all_en_re_rev = {v: k for k, v in all_en_re.items()}
    all_ty_dict_rev = {v: k for k, v in all_ty_dict.items()}
    all_ty_re_rev = {v: k for k, v in all_ty_re.items()}


    true_tail_en_en = OrderedDict()
    true_head_en_en = OrderedDict()
    #  从 yago_insnet_mini.txt中 提true_tail_en_en ，true_head_en_en
    with open('/Users/bubuying/PycharmProjects/DG/data/yago/yago_insnet_mini.txt') as fin:
        for line in fin:
            en1, r, en2 = line.strip().split('\t')
            en1_id = all_en_dict_rev[en1]
            r_id = all_en_re_rev[r]
            en2_id = all_en_dict_rev[en2]
            if (en1_id, r_id) not in true_tail_en_en.keys():
                true_tail_en_en[(en1_id, r_id)] = list()
            true_tail_en_en[(en1_id, r_id)].append(en2_id)
            if (r_id, en2_id) not in true_head_en_en.keys():
                true_head_en_en[(r_id, en2_id)] = list()
            true_head_en_en[(r_id, en2_id)].append(en1_id)




    true_tail_ty_ty = OrderedDict()
    true_head_ty_ty = OrderedDict()
    # 从yago_ontonet.txt中 提true_tail_ty_ty， true_head_ty_ty
    with open('/Users/bubuying/PycharmProjects/DG/data/yago/yago_ontonet.txt') as fin:
        for line in fin:
            ty1, r, ty2 = line.strip().split('\t')
            ty1_id = all_ty_dict_rev[ty1]
            r_id = all_ty_re_rev[r]
            ty2_id = all_ty_dict_rev[ty2]
            if (ty1_id, r_id) not in true_tail_ty_ty.keys():
                true_tail_ty_ty[(ty1_id, r_id)] = list()
            true_tail_ty_ty[(ty1_id, r_id)].append(ty2_id)
            if (r_id, ty2_id) not in true_head_ty_ty.keys():
                true_head_ty_ty[(r_id, ty2_id)] = list()
            true_head_ty_ty[(r_id, ty2_id)].append(ty1_id)



    list_type_G1_sub2_and_sub3 = list()
    list_entity_G1_sub2_and_sub3 = list()

    true_tail_en_ty = OrderedDict()
    true_head_en_ty = OrderedDict()
    # yago_InsType_mini.txt中 提true_tail_en_ty， true_head_en_ty
    with open('/Users/bubuying/PycharmProjects/DG/data/yago/yago_InsType_mini.txt') as fin:
        for line in fin:
            en, _, ty = line.strip().split('\t')
            en_id = all_en_dict_rev[en]
            ty_id = all_ty_dict_rev[ty]
            if en_id not in true_tail_en_ty.keys():
                true_tail_en_ty[en_id] = list()
            true_tail_en_ty[en_id].append(ty_id)
            if ty_id not in true_head_en_ty.keys():
                true_head_en_ty[ty_id] = list()
            true_head_en_ty[ty_id].append(en_id)
            if en_id not in list_entity_G1_sub2_and_sub3:
                list_entity_G1_sub2_and_sub3.append(en_id)
            if ty_id not in list_type_G1_sub2_and_sub3:
                list_type_G1_sub2_and_sub3.append(ty_id)


    return all_graph, all_adj, G2_graph, G2_adj, G1_graph_sub1, G1_sub1_adj, G1_graph_sub2, G1_graph_sub3, G2_links, G1_sub1_links, true_tail_en_en, true_head_en_en, true_tail_ty_ty, true_head_ty_ty, list_type_G1_sub2_and_sub3, true_tail_en_ty, list_entity_G1_sub2_and_sub3, true_head_en_ty



