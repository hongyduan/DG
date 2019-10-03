

import numpy as np
import random

def negative_sample_tail_fun(negative_sample_number, begin_idx, end_idx, true_tail, sample_list_entity_couple, sample_list_id, inter_index, all_node_embedding, embedding_dim):
    # 128 tail negative sample in 1 positive sample
    negative_sample_list = []
    negative_sample_size = 0
    tail_ndarray_negative = np.zeros((1, negative_sample_number, embedding_dim))  # 1*128*500
    while negative_sample_size < negative_sample_number:
        # numpy.random.randint(low, high, size, dtype)  返回随机整数或整型数组，范围区间为[low,high),high没有填写时，默认生成随机数的范围是[0，low)
        negative_sample = random.sample(range(begin_idx, end_idx), negative_sample_number * 2)
        mask_tail = np.in1d(
            negative_sample,
            true_tail[(sample_list_entity_couple[inter_index][0], sample_list_id[inter_index])],
            assume_unique=True,
            invert=True
        )
        negative_sample = negative_sample[mask_tail]
        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size
    negative_sample_tail = np.concatenate(negative_sample_list)[:negative_sample_number]
    for j in range(len(negative_sample_tail)):
        tail_ndarray_negative[1, j, :] = all_node_embedding[negative_sample_tail[j]] # 1*128*500
    return tail_ndarray_negative


def negative_sample_head_fun(negative_sample_number,begin_idx, end_idx, true_head, sample_list_entity_couple, sample_list_id, inter_index, all_node_embedding, embedding_dim):
    # 128 head negative sample in 1 positive sample
    negative_sample_list = []
    negative_sample_size = 0
    head_ndarray_negative = np.zeros((1, negative_sample_number, embedding_dim))  # 1*128*500
    while negative_sample_size < negative_sample_number:
        # numpy.random.randint(low, high, size, dtype)  返回随机整数或整型数组，范围区间为[low,high),high没有填写时，默认生成随机数的范围是[0，low)
        negative_sample = random.sample(range(begin_idx, end_idx), negative_sample_number * 2)
        mask_head = np.in1d(
            negative_sample,
            true_head[(sample_list_id[inter_index], sample_list_entity_couple[inter_index][1])],
            assume_unique=True,
            invert=True
        )
        negative_sample = negative_sample[mask_head]
        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size
    negative_sample_head = np.concatenate(negative_sample_list)[:negative_sample_number]

    for j in range(len(negative_sample_head)):
        head_ndarray_negative[1, j, :] = all_node_embedding[negative_sample_head[j]]  # 1*128*500
    return head_ndarray_negative

def negative_sample_tail_fun_en_ty(negative_sample_number, embedding_dim, list_type_G1_sub2_and_sub3, true_tail_en_ty, inter_index, all_node_embedding):
    size = 0
    list = []
    tail_ndarray_negative = np.zeros((1, negative_sample_number, embedding_dim))  # 1*(128/4)*500
    while size < int(negative_sample_number / 4):
        # random.randint()方法里面的取值区间是前闭后闭区间: [ ]
        # 而np.random.randint()方法的取值区间是前闭后开区间: [ )
        tt = random.sample(list_type_G1_sub2_and_sub3, int(negative_sample_number / 4))
        mask_tail = np.in1d(
            tt,
            true_tail_en_ty[inter_index],
            assume_unique=True,
            invert=True
        )
        negative_sample = tt[mask_tail]
        list.append(negative_sample)
        size += negative_sample.size
    negative_sample_tail = np.concatenate(list)[:negative_sample_number / 4]
    tail_ndarray_negative[1, :, :] = all_node_embedding[negative_sample_tail]

    return tail_ndarray_negative

def negative_sample_head_fun_en_ty(negative_sample_number, embedding_dim, list_entity_G1_sub2_and_sub3, true_head_en_ty, inter_index, all_node_embedding):
    size = 0
    list = []
    head_ndarray_negative = np.zeros((1, negative_sample_number, embedding_dim))  # 1*(128/4)*500
    while size < int(negative_sample_number / 4):
        tt = random.sample(list_entity_G1_sub2_and_sub3, int(negative_sample_number / 4))
        mask_head = np.in1d(
            tt,
            true_head_en_ty[inter_index],
            assume_unique=True,
            invert=True
        )
        negative_sample = tt[mask_head]
        list.append(negative_sample)
        size += negative_sample.size
    negative_sample_head = np.concatenate(list)[:negative_sample_number / 4]
    head_ndarray_negative[1, :, :] = all_node_embedding[negative_sample_head]
    return head_ndarray_negative