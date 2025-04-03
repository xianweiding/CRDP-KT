import math
import os
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#处理数据
class DataProcess():
    def __init__(self, data_folder='assist09', file_name='skill_builder_data_corrected_collapsed.csv', min_inter_num=3):
        print("Process Dataset %s" % data_folder)
        self.min_inter_num = min_inter_num
        self.data_folder = data_folder
        self.file_name = file_name
    #处理csv文件
    def process_csv(self):
        """
            pre-process original csv file for assist dataset
        """

        # read csv file
        data_path = os.path.join(self.data_folder, self.file_name)
        df = pd.read_csv(data_path, low_memory=False, encoding="ISO-8859-1")
        print('original records number %d' % len(df))

        # delete empty skill_id
        df = df.dropna(subset=['skill_id'])
        df = df[~df['skill_id'].isin(['noskill'])]
        print('After removing empty skill_id, records number %d' % len(df))

        # delete scaffolding problems
        df = df[df['original'].isin([1])]
        print('After removing scaffolding problems, records number %d' % len(df))

        #delete the users whose interaction number is less than min_inter_num
        users = df.groupby(['user_id'], as_index=True)
        delete_users = []
        for u in users:
            if len(u[1]) < self.min_inter_num:
                delete_users.append(u[0])
        print('deleted user number based min-inters %d' % len(delete_users))
        df = df[~df['user_id'].isin(delete_users)]
        print('After deleting some users, records number %d' % len(df))
        # print('features: ', df['assistment_id'].unique(), df['answer_type'].unique())

        df.to_csv(os.path.join(self.data_folder, '%s_processed.csv'%self.data_folder))

    #问题-技能的关系图
    def pro_skill_graph(self):
        df = pd.read_csv(os.path.join(self.data_folder, '%s_processed.csv'%self.data_folder),low_memory=False, encoding="ISO-8859-1")
        problems = df['problem_id'].unique()
        pro_id_dict = dict(zip(problems, range(len(problems))))
        print('problem number %d' % len(problems))

        pro_type = df['answer_type'].unique()
        pro_type_dict = dict(zip(pro_type, range(len(pro_type))))
        print('problem type: ', pro_type_dict)

        pro_feat = []
        pro_skill_adj = []
        skill_id_dict, skill_cnt = {}, 0
        all_student_s=[]
        for pro_id in range(len(problems)):            
            tmp_df = df[df['problem_id']==problems[pro_id]]
            tmp_df_0 = tmp_df.iloc[0]#默认获取整行
            # pro_feature: [ms_of_response, answer_type, mean_correct_num]
            ms = tmp_df['ms_first_response'].abs().mean()
            p = tmp_df['correct'].mean()
            pro_type_id = pro_type_dict[tmp_df_0['answer_type']] 
            tmp_pro_feat = [0.] * (len(pro_type_dict)+2)
            tmp_pro_feat[0] = ms
            tmp_pro_feat[pro_type_id+1] = 1.
            tmp_pro_feat[-1] = p
            pro_feat.append(tmp_pro_feat)
            # build problem-skill bipartite 建立问题技能二部图
            tmp_skills = [ele for ele in tmp_df_0['skill_id'].split('_')]
            for s in tmp_skills:
                if s not in skill_id_dict:
                    skill_id_dict[s] = skill_cnt
                    skill_cnt += 1
                pro_skill_adj.append([pro_id, skill_id_dict[s], 1])

        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        pro_feat = np.array(pro_feat).astype(np.float32)
        pro_feat[:, 0] = (pro_feat[:, 0] - np.min(pro_feat[:, 0])) / (np.max(pro_feat[:, 0])-np.min(pro_feat[:, 0]))
        pro_num = np.max(pro_skill_adj[:, 0]) + 1
        skill_num = np.max(pro_skill_adj[:, 1]) + 1
        print('problem number %d, skill number %d' % (pro_num, skill_num))

        # save pro-skill-graph in sparse matrix form
        #sparse.coo_matrix()建立坐标格式的稀疏矩阵
        pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])), shape=(pro_num, skill_num))
        sparse.save_npz(os.path.join(self.data_folder, 'pro_skill_sparse.npz'), pro_skill_sparse)#使用.npz格式将稀疏矩阵保存到文件中

        # take joint skill as a new skill
        skills = df['skill_id'].unique()
        for s in skills:
            if '_' in s:
                skill_id_dict[s] = skill_cnt
                skill_cnt += 1 

        # save pro-id-dict, skill-id-dict
        self.save_dict(pro_id_dict, os.path.join(self.data_folder, 'pro_id_dict.txt'))
        self.save_dict(skill_id_dict, os.path.join(self.data_folder, 'skill_id_dict.txt'))

        # save pro_feat_arr
        np.savez(os.path.join(self.data_folder, 'pro_feat.npz'), pro_feat=pro_feat)

    def pro_skill_graph1(self):
        df = pd.read_csv(os.path.join(self.data_folder, '%s_processed.csv' % self.data_folder), low_memory=False,
                         encoding="ISO-8859-1")
        problems = df['problem_id'].unique()
        pro_id_dict = dict(zip(problems, range(len(problems))))
        print('problem number %d' % len(problems))

        pro_type = df['answer_type'].unique()
        pro_type_dict = dict(zip(pro_type, range(len(pro_type))))
        print('problem type: ', pro_type_dict)

        pro_feat = []
        pro_skill_adj = []
        skill_id_dict, skill_cnt = {}, 0
        all_student_s = []
        for pro_id in range(len(problems)):
            tmp_df = df[df['problem_id'] == problems[pro_id]]
            tmp_df_0 = tmp_df.iloc[0]  # 默认获取整行
            # pro_feature: [ms_of_response, answer_type, mean_correct_num]
            ms = tmp_df['ms_first_response'].abs().mean()
            p = tmp_df['correct'].mean()
            pro_type_id = pro_type_dict[tmp_df_0['answer_type']]
            tmp_pro_feat = [0.] * (len(pro_type_dict) + 2)
            tmp_pro_feat[0] = ms
            tmp_pro_feat[pro_type_id + 1] = 1.
            tmp_pro_feat[-1] = p
            pro_feat.append(tmp_pro_feat)
            # 伪代码编写界
            if tmp_df['correct'].mean() == 0:
                d = 5
            else:
                d = int(1 / (tmp_df['correct'].mean()))
            if d<=4:
                adj=1
            else:
                adj=2
            print(adj)
            # 伪代码编写界
            # build problem-skill bipartite 建立问题技能二部图
            tmp_skills = [ele for ele in tmp_df_0['skill_id'].split('_')]
            for s in tmp_skills:
                if s not in skill_id_dict:
                    skill_id_dict[s] = skill_cnt
                    skill_cnt += 1
                pro_skill_adj.append([pro_id, skill_id_dict[s], adj])

        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        pro_feat = np.array(pro_feat).astype(np.float32)
        pro_feat[:, 0] = (pro_feat[:, 0] - np.min(pro_feat[:, 0])) / (np.max(pro_feat[:, 0]) - np.min(pro_feat[:, 0]))
        pro_num = np.max(pro_skill_adj[:, 0]) + 1
        skill_num = np.max(pro_skill_adj[:, 1]) + 1
        print('problem number %d, skill number %d' % (pro_num, skill_num))


        # save pro-skill-graph in sparse matrix form
        # sparse.coo_matrix()建立坐标格式的稀疏矩阵
        pro_skill_sparse = sparse.coo_matrix(
            (pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])),
            shape=(pro_num, skill_num))
        sparse.save_npz(os.path.join(self.data_folder, 'pro_skill_sparse1.npz'), pro_skill_sparse)  # 使用.npz格式将稀疏矩阵保存到文件中

    #产生用户交互序列，并写入数据文件data.txt
    def generate_user_sequence(self, seq_file):
        # generate user interaction sequence
        # and write to data.txt

        df = pd.read_csv(os.path.join(self.data_folder, '%s_processed.csv'%self.data_folder), low_memory=False, encoding="ISO-8859-1")
        # 伪代码编写界
        all_dff = []
        problems = df['problem_id'].unique()
        for pro_id in range(len(problems)):
            dff = []
            tmp_df = df[df['problem_id'] == problems[pro_id]]  # 取和problems中的id相同的列表元组。
            dff.append(problems[pro_id])
            print(tmp_df['correct'].mean())
            if tmp_df['correct'].mean() == 0:
                dff.append(10)
                all_dff.append(dff)
                print(10)
                continue
            print(1 / (tmp_df['correct'].mean()))
            dff.append(1 / (tmp_df['correct'].mean()))
            all_dff.append(dff)
        # 伪代码编写界
        ui_df = df.groupby(['user_id'], as_index=True)
        print('user number %d' % len(ui_df))

        user_inters = []
        cnt = 0
        for ui in ui_df:
            tmp_user, tmp_inter = ui[0], ui[1]
            tmp_problems = list(tmp_inter['problem_id'])
            tmp_skills = list(tmp_inter['skill_id'])
            tmp_ans = list(tmp_inter['correct'])

            # 伪代码编写界
            Q = []
            for qi, qc, qk in zip(tmp_problems, tmp_ans, tmp_skills):
                q = []
                qd=0
                for dff_arr in all_dff:
                    if qi == dff_arr[0]:
                        qd = dff_arr[1]
                        break
                q.append(qi)
                q.append(qc)
                q.append(qd)
                q.append(qk.split('_'))  # 使用参数作为分割线将原始字符串并组织成列表
                Q.append(q)

            p= math.ceil(len(Q)*0.3)
            # p = 50
            # n = len(Q) // p
            # u = len(Q) % p
            # if u != 0:
            #     m = n + 1
            # else:
            #     m = n
            #print(m)
            for i in range(1):
                # if i + 1 > n:
                #     Q1 = Q[p * i:p * i +u]
                # print(p)
                # else:
                Q1 = Q[p* i:p * (i + 1)]
                QA = []
                for q in Q1:
                    for qa in QA:
                        if qa[1] != q[1]:
                            if qa[1] == 1 and abs(qa[2] - q[2]) >= 0.8 and qa[2] > q[2] and qa[3] == q[3]:
                                if Q1.index(q) - QA.index(qa) != 0:
                                    L = Q1.index(q) - QA.index(qa)
                                else:
                                    L = 1
                                skn = 0
                                c1 = 0
                                c2 = 0
                                QL = Q1[QA.index(qa):Q1.index(q)]
                                for ql in QL:
                                    c2 = c2 + ql[1]
                                    if qa[3] == ql[3]:
                                        c1 = c1 + ql[1]
                                        skn = skn + 1
                                yes = (c1 / (skn + 0.01) + c2 / L + (skn / L))
                                if yes <= 2.8:
                                    tmp_ans[QA.index(qa) + p * i] = 0
                                    QA[QA.index(qa)][1] = 0
                                    print('猜测')
                            elif qa[1] == 0 and abs(qa[2] - q[2]) >= 0.8 and qa[2] < q[2] and qa[3] == q[3]:
                                if Q1.index(q) - QA.index(qa) != 0:
                                    L = Q1.index(q) - QA.index(qa)
                                else:
                                    L = 1
                                skn = 0
                                c1 = 0
                                c2 = 0
                                QL = Q1[QA.index(qa):Q1.index(q)]
                                for ql in QL:
                                    c2 = c2 + ql[1]
                                    if qa[3] == ql[3]:
                                        c1 = c1 + ql[1]
                                        skn = skn + 1
                                yes = (c1 / (skn + 0.01) + c2 / L + (skn / L))
                                if yes >= 0.7:
                                    tmp_ans[QA.index(qa) + p * i] = 1
                                    QA[QA.index(qa)][1] = 1
                                    print('滑倒')
                            # elif qa[1] == 0 and Q1.index(q) - QA.index(qa) >= 7 and qa[3] == q[3]:
                            #     L = Q1.index(q) - QA.index(qa)
                            #     skn = 0
                            #     c1 = 0
                            #     c2 = 0
                            #     QL = Q1[QA.index(qa):Q1.index(q)]
                            #     for ql in QL:
                            #         c2 = c2 + ql[1]
                            #         if qa[3] == ql[3]:
                            #             c1 = c1 + ql[1]
                            #             skn = skn + 1
                            #     yes = (c1 / (skn + 0.01) + c2 / L + (skn / L))
                            #     if yes > 2.146:
                            #         tmp_ans[QA.index(qa) + p * i] = 1
                            #         QA[QA.index(qa)][1] = 1
                            #         print('正确连续性')
                            # elif qa[1] == 1 and Q1.index(q) - QA.index(qa) >= 7 and qa[3] == q[3]:
                            #     L = Q1.index(q) - QA.index(qa)
                            #     skn = 0
                            #     c1 = 0
                            #     c2 = 0
                            #     QL = Q1[QA.index(qa):Q1.index(q)]
                            #     for ql in QL:
                            #         c2 = c2 + ql[1]
                            #         if qa[3] == ql[3]:
                            #             c1 = c1 + ql[1]
                            #             skn = skn + 1
                            #     yes = (c1 / (skn + 0.01) + c2 / L + (skn / L))
                            #     if yes < 0.9:
                            #         tmp_ans[QA.index(qa) + p * i] = 0
                            #         QA[QA.index(qa)][1] = 0
                            #         print('错误连续性')
                    QA.append(q)
            user_inters.append([[len(tmp_inter)], tmp_skills, tmp_problems, tmp_ans])
        pro_skill_adj = []
        skill_id_dict, skill_cnt = {}, 0
        for pro_id in range(len(problems)):
            tmp_df = df[df['problem_id'] == problems[pro_id]]
            tmp_df_0 = tmp_df.iloc[0]  # 默认获取整行

            # build problem-skill bipartite 建立问题技能二部图
            tmp_skills = [ele for ele in tmp_df_0['skill_id'].split('_')]
            for s in tmp_skills:
                if s not in skill_id_dict:
                    skill_id_dict[s] = skill_cnt
                    skill_cnt += 1
                pro_skill_adj.append([pro_id, skill_id_dict[s], 1])

        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        # save pro-skill-graph in sparse matrix form
        # sparse.coo_matrix()建立坐标格式的稀疏矩阵
        # pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])),shape=(pro_num, skill_num))
        # sparse.save_npz(os.path.join(self.data_folder, 'pro_skill_sparse.npz'),pro_skill_sparse)  # 使用.npz格式将稀疏矩阵保存到文件中

            # 伪代码编写界
        write_file = os.path.join(self.data_folder, seq_file)
        self.write_txt(write_file, user_inters)

        # 产生用户交互序列，并写入数据文件data.txt




    #保存dict
    def save_dict(self, dict_name, file_name):
        f = open(file_name, 'w')
        f.write(str(dict_name))
        f.close

    #写入txt文件
    def write_txt(self, file, data):
        with open(file, 'w') as f:
            for dd in data:
                for d in dd:
                    f.write(str(d)+'\n')

    #读取用户序列
    def read_user_sequence(self, filename, max_len=200, min_len=3, shuffle_flag=True):
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(os.path.join(self.data_folder, 'skill_id_dict.txt'), 'r') as f:
            skill_id_dict = eval(f.read()) 
        with open(os.path.join(self.data_folder, 'pro_id_dict.txt'), 'r') as f:
            pro_id_dict = eval(f.read())
        

        y, skill, problem, real_len = [], [], [], []
        skill_num, pro_num = len(skill_id_dict), len(pro_id_dict)
        print('skill num, pro num, ', skill_num, pro_num)

        index = 0
        while index < len(lines):
            num = eval(lines[index])[0]
            tmp_skills = eval(lines[index+1])[:max_len]
            tmp_skills = [skill_id_dict[ele]+1 for ele in tmp_skills]     # for assist09
            # tmp_skills = [ele+1 for ele in tmp_skills]                      # for assist12 
            tmp_pro = eval(lines[index+2])[:max_len]
            tmp_pro = [pro_id_dict[ele]+1 for ele in tmp_pro]
            tmp_ans = eval(lines[index+3])[:max_len]

            if num>=min_len:
                tmp_real_len = len(tmp_skills)
                # Completion sequence
                tmp_ans += [-1]*(max_len-tmp_real_len)
                tmp_skills += [0]*(max_len-tmp_real_len)
                tmp_pro += [0]*(max_len-tmp_real_len)



                y.append(tmp_ans)
                skill.append(tmp_skills)
                problem.append(tmp_pro)
                real_len.append(tmp_real_len)

            index += 4
        
        y = np.array(y).astype(np.float32)
        skill = np.array(skill).astype(np.int32)
        problem = np.array(problem).astype(np.int32)
        real_len = np.array(real_len).astype(np.int32)

        print(skill.shape, problem.shape, y.shape, real_len.shape)      
        print(np.max(y), np.min(y))
        print(np.max(real_len), np.min(real_len))  
        print(np.max(skill), np.min(skill))
        print(np.max(problem), np.min(problem))

        np.savez(os.path.join("assist09", "assist091.npz"), problem=problem, y=y, skill=skill, real_len=real_len, skill_num=skill_num, problem_num=pro_num)

    def read_user_sequence1(self, filename, max_len=200, min_len=3, shuffle_flag=True):
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(os.path.join(self.data_folder, 'skill_id_dict.txt'), 'r') as f:
            skill_id_dict = eval(f.read())
        with open(os.path.join(self.data_folder, 'pro_id_dict.txt'), 'r') as f:
            pro_id_dict = eval(f.read())

        y, skill, problem, real_len = [], [], [], []
        skill_num, pro_num = len(skill_id_dict), len(pro_id_dict)
        print('skill num, pro num, ', skill_num, pro_num)

        index = 0
        while index < len(lines):
            num = eval(lines[index])[0]
            tmp_skills = eval(lines[index + 1])[:max_len]
            tmp_skills = [skill_id_dict[ele] + 1 for ele in tmp_skills]  # for assist09
            # tmp_skills = [ele+1 for ele in tmp_skills]                      # for assist12
            tmp_pro = eval(lines[index + 2])[:max_len]
            tmp_pro = [pro_id_dict[ele] + 1 for ele in tmp_pro]
            tmp_ans = eval(lines[index + 3])[:max_len]

            if num >= min_len:
                tmp_real_len = len(tmp_skills)
                # Completion sequence
                tmp_ans += [-1] * (max_len - tmp_real_len)
                tmp_skills += [0] * (max_len - tmp_real_len)
                tmp_pro += [0] * (max_len - tmp_real_len)

                y.append(tmp_ans)
                skill.append(tmp_skills)
                problem.append(tmp_pro)
                real_len.append(tmp_real_len)

            index += 4

        y = np.array(y).astype(np.float32)
        skill = np.array(skill).astype(np.int32)
        problem = np.array(problem).astype(np.int32)
        real_len = np.array(real_len).astype(np.int32)

        print(skill.shape, problem.shape, y.shape, real_len.shape)
        print(np.max(y), np.min(y))
        print(np.max(real_len), np.min(real_len))
        print(np.max(skill), np.min(skill))
        print(np.max(problem), np.min(problem))

        np.savez(os.path.join(self.data_folder, "%s.npz" % 'assist091'), problem=problem, y=y, skill=skill,
                 real_len=real_len, skill_num=skill_num, problem_num=pro_num)





if __name__ == '__main__':
    data_folder = 'assist09'
    min_inter_num = 3
    file_name='skill_builder_data_corrected_collapsed.csv'

    DP = DataProcess(data_folder, file_name, min_inter_num)

    ## excute the following function step by step
    # DP.process_csv()
    # DP.pro_skill_graph()
    # DP.pro_skill_graph1()
    DP.generate_user_sequence('dataf1-test.txt')
    # DP.generate_user_sequence1('data2.txt')
    # DP.generate_user_sequence2()
    # DP.read_user_sequence(os.path.join(data_folder, 'data1.txt'))
    # DP.generate_user_sequence_weight_matrix()
    # DP.read_user_sequence1(os.path.join(data_folder, 'data1.txt'))
    # DP.read_user_sequence2(os.path.join(data_folder, 'data2.txt'))
    # DP.process_ednet(os.path.join('ednet', 'data1.txt'))




