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
            # 以下伪代码将融入观点
            QA = []
            for q in Q:
                for qa in QA:
                    if qa[1]!=q[1]:
                        if qa[1]==1 and abs(qa[2]-q[2])>=0.5 and qa[2]>q[2] and qa[3]==q[3]:
                            tmp_ans[QA.index(qa)]=0
                            QA[QA.index(qa)][1]=0
                            print('偶发')
                        elif qa[1]==0 and abs(qa[2]-q[2])>=0.5 and qa[2]<q[2] and qa[3]==q[3]:
                            tmp_ans[QA.index(qa)]=1
                            QA[QA.index(qa)][1] = 1
                            print('进步')
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

    def generate_user_sequence1(self, seq_file):
        # generate user interaction sequence
        # and write to data.txt

        df = pd.read_csv(os.path.join(self.data_folder, '%s_processed.csv' % self.data_folder), low_memory=False,
                         encoding="ISO-8859-1")
        # 伪代码编写界
        all_dff = []
        problems = df['problem_id'].unique()
        for pro_id in range(len(problems)):
            dff = []
            tmp_df = df[df['problem_id'] == problems[pro_id]]  # 取和problems中的id相同的列表元组。
            dff.append(problems[pro_id])
            # print(tmp_df['correct'].mean())
            if tmp_df['correct'].mean() == 0:
                dff.append(10)
                all_dff.append(dff)
                # print(10)
                continue
            # print(1 / (tmp_df['correct'].mean()))
            dff.append(1 / (tmp_df['correct'].mean()))
            all_dff.append(dff)
        # 伪代码编写界
        ui_df = df.groupby(['user_id'], as_index=True)
        print('user number %d' % len(ui_df))

        user_inters = []
        cnt = 0
        print("starting processing")
        for ui in tqdm(ui_df):
            tmp_user, tmp_inter = ui[0], ui[1]
            tmp_problems = list(tmp_inter['problem_id'])
            tmp_skills = list(tmp_inter['skill_id'])
            tmp_ans = list(tmp_inter['correct'])

            # 伪代码编写界
            Q = []
            for qi, qc, qk in zip(tmp_problems, tmp_ans, tmp_skills):
                q = []
                qd = 0
                for dff_arr in all_dff:
                    if qi == dff_arr[0]:
                        qd = dff_arr[1]
                        break
                q.append(qi)
                q.append(qc)
                q.append(qd)
                q.append(qk.split('_'))  # 使用参数作为分割线将原始字符串并组织成列表
                Q.append(q)
            # 以下伪代码将融入观点
            QA = []
            for q in Q:
                QA.append(q)
                qH=QA
                for qa in QA:
                    sumc=0
                    sumd=0
                    for qh in qH:
                        if qa[3]==qh[3] and abs(qa[2]-qh[2])<0.3:
                            d=abs(qH.index(qh)-QA.index(qa))
                            if d==0:
                                d=1
                            sumc += (qh[2] * qh[1])/d
                            sumd += qh[2]/d
                        else:
                              continue
                    if sumd:
                        cf=sumc/sumd
                        # print(cf)
                        if cf>=0.5:
                            tmp_ans[QA.index(qa)]=1
                            QA[QA.index(qa)][1]=1
                        else:
                            tmp_ans[QA.index(qa)] = 0
                            QA[QA.index(qa)][1] = 0

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

    def generate_user_sequence2(self):

        with open('dataf1.txt', 'r') as f:
            lines = f.readlines()
        index = 0
        num = eval(lines[index])[0]
        tmp_skills = eval(lines[index + 1])
        tmp_pro = eval(lines[index + 2])
        tmp_ans = eval(lines[index + 3])
        print(len(lines) / 4)
        all_user = []
        k = 0
        for i in range(0, int(len(lines) / 4)):
            user = []
            user.append(eval(lines[k])[0])
            user.append(eval(lines[k + 1]))
            user.append(eval(lines[k + 2]))
            user.append(eval(lines[k + 3]))
            print(user)
            all_user.append(user)
            print('---------')
            k = k + 4
        print('finish')
        df = pd.read_csv(os.path.join('%s_processed.csv' % self.data_folder), low_memory=False,
                         encoding="ISO-8859-1")
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
        user_inters = []

        for ui in tqdm(range(0, len(all_user))):
            user = all_user[ui]
            tmp_problems = user[2]
            tmp_skills = user[1]
            tmp_ans = user[3]

            # 伪代码编写界
            Q = []
            for qi, qc, qk in zip(tmp_problems, tmp_ans, tmp_skills):
                q = []
                qd = 0
                for dff_arr in all_dff:
                    if qi == dff_arr[0]:
                        qd = dff_arr[1]
                        break
                q.append(qi)
                q.append(qc)
                q.append(qd)
                q.append(qk.split('_'))  # 使用参数作为分割线将原始字符串并组织成列表
                Q.append(q)
            QA = []
            for q in Q:
                QA.append(q)
                qH = QA
                for qa in QA:
                    sumc = 0
                    sumd = 0
                    for qh in qH:
                        if qa[3] == qh[3] and abs(qa[2] - qh[2]) < 0.05:
                            p = QA.index(qa) - qH.index(qh)
                            if p > 0:
                                d = abs(p)
                            elif p < 0:
                                d = 1
                            if p == 0:
                                d = 1
                            sumc += (qh[2] * qh[1]) / d
                            sumd += qh[2] / d
                        else:
                            continue
                    if sumd:
                        cf = sumc / sumd
                        # print(cf)
                        if cf >= 0.5:
                            tmp_ans[QA.index(qa)] = 1
                            QA[QA.index(qa)][1] = 1
                        else:
                            tmp_ans[QA.index(qa)] = 0
                            QA[QA.index(qa)][1] = 0
            user_inters.append([[len(tmp_ans)], tmp_skills, tmp_problems, tmp_ans])

        # 伪代码编写界
        with open('dataf3.txt', 'w') as f:
            for dd in user_inters:
                for d in dd:
                    f.write(str(d) + '\n')
        # write_file = os.path.join('ednet', 'data1.txt')
        # self.write_txt(write_file, user_inters)

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

    def read_user_sequence2(self, filename, max_len=200, min_len=3, shuffle_flag=True):
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

        np.savez(os.path.join(self.data_folder, "%s.npz" % 'assist092'), problem=problem, y=y, skill=skill,
                 real_len=real_len, skill_num=skill_num, problem_num=pro_num)

    def read_user_sequence3(self, filename, max_len=200, min_len=3, shuffle_flag=True):
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

        np.savez(os.path.join(self.data_folder, "%s.npz" % 'assist093'), problem=problem, y=y, skill=skill,
                 real_len=real_len, skill_num=skill_num, problem_num=pro_num)

    def generate_user_sequence_weight_matrix(self):

        df = pd.read_csv(os.path.join(self.data_folder, '%s_processed.csv'%self.data_folder), low_memory=False, encoding="ISO-8859-1")
        # 伪代码编写界
        all_dff = []
        problems = df['problem_id'].unique()
        for pro_id in range(len(problems)):
            dff = []
            tmp_df = df[df['problem_id'] == problems[pro_id]]  # 取和problems中的id相同的列表元组。
            dff.append(problems[pro_id])
            print(tmp_df['correct'].mean())
            if tmp_df['correct'].mean()==0:
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
        all_student_s=[]
        for ui in ui_df:
            tmp_user, tmp_inter = ui[0], ui[1]
            tmp_problems = list(tmp_inter['problem_id'])
            tmp_skills = list(tmp_inter['skill_id'])
            tmp_ans = list(tmp_inter['correct'])
            user_inters.append([[len(tmp_inter)], tmp_skills, tmp_problems, tmp_ans])
            # 伪代码编写界
            Q = []
            for qi, qc, qk in zip(tmp_problems, tmp_ans,tmp_skills):
                q = []
                q_w_s = []
                qd=0
                for dff_arr in all_dff:
                    if qi == dff_arr[0]:
                        qd = dff_arr[1]
                        print(qd)
                        break
                q.append(qi)
                q.append(qc)
                q.append(qd)
                q.append(qk.split('_'))  # 使用参数作为分割线将原始字符串并组织成列表
                for i in range(0,len(qk.split('_'))):
                    q_w_s.append(1)
                q.append(q_w_s)
                Q.append(q)
            # 以下伪代码将融入观点
            QA = []
            for q in Q:
                    for qa in QA:
                        a = abs(q[2] - qa[2])
                        print(a)
                        b = set(q[3]).issubset(set(qa[3])) or set(qa[3]).issubset(set(q[3]))
                        print(q[3],qa[3])
                        print(q[4],qa[4])
                        print(q[1],qa[1])
                        print(b)
                        # issubset是判断列表是否是另一个列表的子集
                        if q[1] != qa[1] and a <= 1.5 and b:
                            if len(q[3])>len(qa[3]):
                                c=q[3]
                            else:
                                c=qa[3]
                            for str_s in c:
                                if str_s not in list(set(qa[3]) ^ set(q[3])):
                                    if (len(q[3]) > len(qa[3])):
                                        Q[Q.index(q)][4][q[3].index(str_s)] = Q[Q.index(q)][4][q[3].index(str_s)] + 1
                                    else:
                                        Q[Q.index(qa)][4][qa[3].index(str_s)] = Q[Q.index(qa)][4][qa[3].index(str_s)] + 1
                    QA.append(q)
            all_student_s.append(Q)

        pro_skill_adj = []
        skill_id_dict, skill_cnt = {}, 0
        for pro_id in range(len(problems)):
            tmp_df = df[df['problem_id'] == problems[pro_id]]
            tmp_df_0 = tmp_df.iloc[0]  # 默认获取整行
            m=[0]*len(tmp_df_0['skill_id'].split('_'))
            for Q in all_student_s:
                for q in Q:
                    if problems[pro_id] == q[0]:
                        m=list(np.add(m,q[4]))
                        print(m)
            sum1=sum(m)
            for i in range(0,len(m)):
              m[i]=m[i]/sum1
            print(m)
            # build problem-skill bipartite 建立问题技能二部图
            tmp_skills = [ele for ele in tmp_df_0['skill_id'].split('_')]
            for s in tmp_skills:
                if s not in skill_id_dict:
                    skill_id_dict[s] = skill_cnt
                    skill_cnt += 1
                L=[pro_id,skill_id_dict[s],m[tmp_skills.index(s)]]
                pro_skill_adj.append(L)

        pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)
        pro_num = np.max(pro_skill_adj[:, 0]) + 1
        skill_num = np.max(pro_skill_adj[:, 1]) + 1
        print('problem number %d, skill number %d' % (pro_num, skill_num))
        # save pro-skill-graph in sparse matrix form
        # sparse.coo_matrix()建立坐标格式的稀疏矩阵
        pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])),shape=(pro_num, skill_num))
        sparse.save_npz(os.path.join(self.data_folder, 'pro_skill_sparse_martix.npz'),pro_skill_sparse)  # 使用.npz格式将稀疏矩阵保存到文件中

    # def process_ednet(self, filename, max_len=200, min_len=3, shuffle_flag=True):
    #     with open(filename, 'r') as f:
    #         lines = f.readlines()
    #     with open(os.path.join('ednet', 'skill_id_dict.txt'), 'r') as f:
    #         skill_id_dict = eval(f.read())
    #     with open(os.path.join('ednet', 'pro_id_dict.txt'), 'r') as f:
    #         pro_id_dict = eval(f.read())
    #
    #     y, skill, problem, real_len = [], [], [], []
    #     skill_num, pro_num = len(skill_id_dict), len(pro_id_dict)
    #     print('skill num, pro num, ', skill_num, pro_num)
    #
    #     index = 0
    #     while index < len(lines):
    #         num = eval(lines[index])[0]
    #         tmp_skills = eval(lines[index + 1])[:max_len]
    #         tmp_skills = [skill_id_dict[ele] + 1 for ele in tmp_skills]  # for assist09
    #         # tmp_skills = [ele+1 for ele in tmp_skills]                      # for assist12
    #         tmp_pro = eval(lines[index + 2])[:max_len]
    #         tmp_pro = [pro_id_dict[ele] + 1 for ele in tmp_pro]
    #         tmp_ans = eval(lines[index + 3])[:max_len]

if __name__ == '__main__':
    data_folder = 'assist09'
    min_inter_num = 3
    file_name='skill_builder_data_corrected_collapsed.csv'

    DP = DataProcess(data_folder, file_name, min_inter_num)

    ## excute the following function step by step
    # DP.process_csv()
    # DP.pro_skill_graph()
    # DP.pro_skill_graph1()
    # DP.generate_user_sequence('data1.txt')
    # DP.generate_user_sequence1('data2.txt')
    DP.generate_user_sequence2()
    # DP.read_user_sequence(os.path.join(data_folder, 'data1.txt'))
    # DP.generate_user_sequence_weight_matrix()
    # DP.read_user_sequence1(os.path.join(data_folder, 'data1.txt'))
    # DP.read_user_sequence2(os.path.join(data_folder, 'data2.txt'))
    # DP.read_user_sequence3(os.path.join(data_folder, 'data3.txt'))
    # DP.process_ednet(os.path.join('ednet', 'data1.txt'))




