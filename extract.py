import os
import sys
import numpy as np
from scipy import sparse
import time

#https://pypi.tuna.tsinghua.edu.cn/simple


data_folder = 'assist12'
pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz'))
[pro_num, skill_num] = pro_skill_coo.toarray().shape
print('problem number %d, skill number %d' % (pro_num, skill_num))
pro_skill_csc = pro_skill_coo.tocsc()#将此数组/矩阵转换为压缩稀疏列格式重复的条目将被汇总在一起。
pro_skill_csr = pro_skill_coo.tocsr()#将此数组/矩阵转换为压缩稀疏行格式重复的条目将被汇总在一起。

#提取问题与问题之间的相似矩阵
def extract_pro_pro_sim():
    # extract pro-pro similarity sparse matrix
    pro_pro_adj = []
    for p in range(pro_num):
        tmp_skills = pro_skill_csr.getrow(p).indices
        similar_pros = pro_skill_csc[:, tmp_skills].indices
        zipped = zip([p] * similar_pros.shape[0], similar_pros)
        pro_pro_adj += list(zipped)

    pro_pro_adj = list(set(pro_pro_adj))
    pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
    data = np.ones(pro_pro_adj.shape[0]).astype(np.float32)
    pro_pro_sparse = sparse.coo_matrix((data, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(pro_num, pro_num))
    sparse.save_npz(os.path.join(data_folder, 'pro_pro_sparse.npz'), pro_pro_sparse)

#提取技能与技能相似矩阵
def extract_skill_skill_sim():
    # extract skill-skill similarity sparse matrix
    skill_skill_adj = []
    for s in range(skill_num):
        tmp_pros = pro_skill_csc.getcol(s).indices
        similar_skills = pro_skill_csr[tmp_pros, :].indices
        zipped = zip([s] * similar_skills.shape[0], similar_skills)
        skill_skill_adj += list(zipped)

    skill_skill_adj = list(set(skill_skill_adj))
    skill_skill_adj = np.array(skill_skill_adj).astype(np.int32)
    data = np.ones(skill_skill_adj.shape[0]).astype(np.float32)
    skill_skill_sparse = sparse.coo_matrix((data, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])), shape=(skill_num, skill_num))
    sparse.save_npz(os.path.join(data_folder, 'skill_skill_sparse.npz'), skill_skill_sparse)

#提取问题与问题之间的相似矩阵
def extract_pro_pro_sim1():
    # extract pro-pro similarity sparse matrix
    pro_pro_adj = []
    for p in range(pro_num):
        tmp_skills = pro_skill_csr.getrow(p).indices
        similar_pros = pro_skill_csc[:, tmp_skills].indices
        zipped = zip([p] * similar_pros.shape[0], similar_pros)
        pro_pro_adj += list(zipped)

    pro_pro_adj = list(set(pro_pro_adj))
    pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
    data = np.ones(pro_pro_adj.shape[0]).astype(np.float32)
    pro_pro_sparse = sparse.coo_matrix((data, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(pro_num, pro_num))
    sparse.save_npz(os.path.join(data_folder, 'pro_pro_sparse1.npz'), pro_pro_sparse)

#提取技能与技能相似矩阵
def extract_skill_skill_sim1():
    # extract skill-skill similarity sparse matrix
    skill_skill_adj = []
    for s in range(skill_num):
        tmp_pros = pro_skill_csc.getcol(s).indices
        similar_skills = pro_skill_csr[tmp_pros, :].indices
        zipped = zip([s] * similar_skills.shape[0], similar_skills)
        skill_skill_adj += list(zipped)

    skill_skill_adj = list(set(skill_skill_adj))
    skill_skill_adj = np.array(skill_skill_adj).astype(np.int32)
    data = np.ones(skill_skill_adj.shape[0]).astype(np.float32)
    skill_skill_sparse = sparse.coo_matrix((data, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])), shape=(skill_num, skill_num))
    sparse.save_npz(os.path.join(data_folder, 'skill_skill_sparse1.npz'), skill_skill_sparse)

extract_pro_pro_sim()
extract_skill_skill_sim()
# extract_pro_pro_sim1()
# extract_skill_skill_sim1()