#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch.nn.functional as F
from model import GCN_DD
from utils import *
import collections
import argparse
import torch
import json



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--init_checkpoint', default=None, type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--embedding_dim', type=int, default=500)

    parser.add_argument('--data_path_bef', type=str, default="/Users/bubuying/PycharmProjects/DGraph/data/yago")
    parser.add_argument('--data_path', type=str, default="/Users/bubuying/PycharmProjects/DGraph/data/yago_result")
    parser.add_argument('--save_path', type=str, default="/Users/bubuying/PycharmProjects/DGraph/save")

    parser.add_argument('--negative_sample_number', type=int, default=128)  # 128
    parser.add_argument('--num_node_G2', type=int, default=0, help='DO NOT MANUALLY SET') # 26078
    parser.add_argument('--num_relation_G2', type=int, default=0, help='DO NOT MANUALLY SET') # 34
    parser.add_argument('--num_relation_G1_sub1', type=int, default=0, help='DO NOT MANUALLY SET')  # 30
    parser.add_argument('--num_all_node_G1_and_G2', type=int, default=0, help='DO NOT MANUALLY SET') # 8948+26078
    parser.add_argument('--num_entity_node_G1', type=int, default=0, help='DO NOT MANUALLY SET') # 8948
    parser.add_argument('--num_type_node_G1', type=int, default=0, help='DO NOT MANUALLY SET') # 911
    parser.add_argument('--num_type_node_G1_sub2_and_sub3', type=int, default=0, help='DO NOT MANUALLY SET') # 106
    parser.add_argument('--sample_num', type=int, default=1000) # 100
    parser.add_argument('--gamma', type=int, default=12)  # 12
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 12
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)

    return parser.parse_args(args)


def override_config(args):
    with open(os.path.join(args.init_checkpoint, 'config.json'),'r') as fjson:
        argparse_dict = json.load(fjson)
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_node_embedding = argparse_dict['double_node_embedding']
    args.double_re_embedding = argparse_dict['double_re_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def set_logger(args):
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test_log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def get_loss(negative_score_tail, adversarial_temperature, negative_score_head, positive_score):
    negative_score_tail = F.softmax(negative_score_tail * adversarial_temperature,
                                      dim=1).detach() * F.logsigmoid(-negative_score_tail)
    negative_score_tail = negative_score_tail.sum(dim=1)  # negative_score 1024*128--->1024
    negative_score_head = F.softmax(negative_score_head * adversarial_temperature,
                                      dim=1).detach() * F.logsigmoid(-negative_score_head)
    negative_score_head = negative_score_head.sum(dim=1)  # negative_score 1024*128--->1024
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)  # positive_score 1024*1--->1024
    positive_sample_loss = - positive_score.mean()  # positive_sample_loss : 1
    negative_sample_loss_tail = - negative_score_tail.mean()  # negative_sample_loss : 1
    negative_sample_loss_head = - negative_score_head.mean()  # negative_sample_loss : 1
    return positive_sample_loss, negative_sample_loss_tail, negative_sample_loss_head



def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('One of train/valid/test mode must be choosed.')
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('One of init_checkpoint/data_path must be choosed.')
    if args.do_train and args.save_path is None:
        raise ValueError('where do you want to save your model?')
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    all_node2id, node2id_G2, type_node2id_G1, entity_node2id_G1, relation2id_G2, relation2id_G1_sub1 = get_dict(args)

    all_graph, all_adj, G2_graph, G2_adj, G1_graph_sub1, G1_sub1_adj, G1_graph_sub2, G1_graph_sub3, G2_links, G1_sub1_links, true_tail_en_en, true_head_en_en, true_tail_ty_ty, true_head_ty_ty, list_type_G1_sub2_and_sub3, true_tail_en_ty, list_entity_G1_sub2_and_sub3, true_head_en_ty = load_data(args, all_node2id, node2id_G2, type_node2id_G1, entity_node2id_G1, relation2id_G1_sub1, relation2id_G2)

    gcn_dd = GCN_DD(G2_adj, args.embedding_dim, args.num_node_G2, G1_sub1_adj, args.num_type_node_G1, G1_graph_sub2, G1_graph_sub3, args.num_entity_node_G1, args.num_type_node_G1_sub2_and_sub3, args.sample_num, args.num_node_G2, G2_links, G1_sub1_links, args.gamma, args.num_relation_G2, args.num_relation_G1_sub1, args.num_type_node_G1, args.num_type_node_G1, args.negative_sample_number, true_tail_en_en, true_head_en_en, true_tail_ty_ty, true_head_ty_ty, list_type_G1_sub2_and_sub3, true_tail_en_ty, list_entity_G1_sub2_and_sub3, true_head_en_ty)

    opti = torch.optim.Adam(gcn_dd.parameters(), lr=args.learning_rate)


    for i in range(1):
    #for i in range(args.epoch):
        opti.zero_grad()
        # positive_score_1: 1024*1;   negative_score_1_tail: 1024*128;   negative_score_1_head: 1024*128;
        # positive_score_2: 1024*1;   negative_score_2_tail: 1024*128;   negative_score_2_head: 1024*128;
        # positive_score_3: 1024*1;   negative_score_3_tail: 1024*32;    negative_score_3_head: 1024*32;
        positive_score_1, negative_score_1_tail, negative_score_1_head, positive_score_2, negative_score_2_tail, negative_score_2_head, positive_score_3, negative_score_3_tail, negative_score_3_head = gcn_dd() # 1000
        positive_sample_loss_1, negative_sample_loss_tail_1, negative_sample_loss_head_1 = get_loss(negative_score_1_tail, args.adversarial_temperature, negative_score_1_head, positive_score_1)
        positive_sample_loss_2, negative_sample_loss_tail_2, negative_sample_loss_head_2 = get_loss(negative_score_2_tail, args.adversarial_temperature, negative_score_2_head, positive_score_2)
        positive_sample_loss_3, negative_sample_loss_tail_3, negative_sample_loss_head_3 = get_loss(negative_score_3_tail, args.adversarial_temperature, negative_score_3_head, positive_score_3)
        loss1 = (positive_sample_loss_1 + negative_sample_loss_tail_1 + negative_sample_loss_head_1)/3
        loss2 = (positive_sample_loss_2 + negative_sample_loss_tail_2 + negative_sample_loss_head_2)/3
        loss3 = (positive_sample_loss_3 + negative_sample_loss_tail_3 + negative_sample_loss_head_3)/3

        loss = (loss1 + loss2 + loss3)/3
        loss.backward()

        opti.step()


def get_dict(args):
    #  G2
    #  entity2id, relation2id分别是两个dict。
    node2id_G2 = OrderedDict()
    node2id_G2_re = OrderedDict()
    relation2id_G2 = OrderedDict()
    relation2id_G1_sub1 = OrderedDict()

    with open(os.path.join(args.data_path, 'final_entity_order.txt')) as fin:  # 26078
        for line in fin:
            eid, entity = line.strip().split('\t')
            node2id_G2[eid] = entity
            node2id_G2_re[entity] = eid

    with open(os.path.join(args.data_path, 'ffinal_en_relation_order.txt')) as fin:  # 34
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id_G2[rid] = relation

    with open(os.path.join(args.data_path, 'ffinal_ty_relation_order.txt')) as fin:  # 30
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id_G1_sub1[rid] = relation

    args.num_node_G2 = len(node2id_G2)  # entity 26078
    args.num_relation_G2 = len(relation2id_G2)  # relation 34
    args.num_relation_G1_sub1 = len(relation2id_G1_sub1)  # relation 30


    #  G1
    type_node2id_G1 = collections.OrderedDict()
    type_node2id_G1_re = collections.OrderedDict() # 911
    entity_node2id_G1 = collections.OrderedDict()  # 8948
    type_node2id_G1_sub2_and_sub3 = collections.OrderedDict()  # 106
    #  读取911个type
    with open(os.path.join(args.data_path, 'final_type_order.txt')) as fin:  # 911
        for line in fin:
            tyid, type = line.strip().split('\t')
            type_node2id_G1[int(tyid) + args.num_node_G2] = type
            type_node2id_G1_re[type] = int(tyid) + args.num_node_G2
    args.num_type_node_G1 = len(type_node2id_G1)  # 911


    # 读取8948个entity
    with open(os.path.join(args.data_path_bef, 'yago_InsType_mini.txt')) as fin: # 8948
        for line in fin:
            en, _, ty = line.strip().split('\t')
            if en not in entity_node2id_G1.keys():
                entity_node2id_G1[node2id_G2_re[en]] = en
            if ty not in type_node2id_G1_sub2_and_sub3.keys():
                type_node2id_G1_sub2_and_sub3[type_node2id_G1_re[ty]] = ty
    args.num_entity_node_G1 = len(entity_node2id_G1)  # 8948
    args.num_type_node_G1_sub2_and_sub3 = len(type_node2id_G1_sub2_and_sub3)
    # all_node: 26078entity + 911type
    all_node2id = node2id_G2.copy()
    all_node2id.update(type_node2id_G1)
    args.num_all_node_G1_and_G2 = len(all_node2id) # 26078 + 911


    return all_node2id, node2id_G2, type_node2id_G1, entity_node2id_G1, relation2id_G2, relation2id_G1_sub1

if __name__ == '__main__':
    main(parse_args())
    print("finished")