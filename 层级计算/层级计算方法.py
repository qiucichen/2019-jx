#!/usr/bin/env python
# coding: utf-8

# # 思路
# ## 推荐算法
# ### 1 数据
# 1、左右结构的数据；\
# 2、上下级数据缺失的数据补全，下级完全不一样\
# 3、展示数据互为推荐，展示数据的重复推荐\
# 4、处理掉互为推荐，重复推荐可以不处理
# ### 2 节点
# 1、chain(*df.values)，构建节点数据 \
# 2、节点无下级的数据和有推荐下级的数据分别处理
# ### 3 构建全层级df
# 1、节点构建df
# 2、构建所有下级
# ### 4 统计发展下限和人数
# 1、分组统计每个节点的发展下限发展人数\
# 注意：nan的处理方式
# ### 5 未发展的节点合并上

# In[1]:


import numpy as np
import pandas as pd
from itertools import chain


# ## 1 数据导入与处理

# In[2]:


data = pd.read_excel(r"数据.xlsx",dtype=np.str,encode='utf-8')


# In[276]:


data1 = data.copy()
data1.rename(columns={"ID":"userid","邀请人ID":"t_userid"},inplace=True)  # 字段名称统一   userid : 直接下线名称，t_userid : 直接上线名称
data1.head()


# In[277]:


data1=data1.append({"userid":"a","t_userid":"a"},ignore_index=True)


# In[278]:


# 数据去重
data1.drop_duplicates(inplace=True)


# In[279]:


# 数据缺失处理
# 直接下线缺失填充上不同的值 wz_i(i=0,1,2,3...) ; 直接上线缺失填充上相同的值 wz_s
data1.loc[data1.userid.isnull(),"userid"] = ["wz_"+str(i) for i in range(len(data1[data1.userid.isnull()]))]
data1.fillna("wz_s",inplace=True)


# In[288]:


#data1 = data1.append({"userid":"a","t_userid":"b"},ignore_index=True)
#data1 = data1.append({"userid":"b","t_userid":"a"},ignore_index=True)


# In[293]:


# 相互推荐数据查找并删除
data_hu_tui = pd.merge(data1,data1,how="left",left_on="t_userid",right_on="userid")
data_hu_tui_1 = data_hu_tui[data_hu_tui.userid_x==data_hu_tui.t_userid_y]         # 相互推荐数据
list_hutui_1 = []
for i in data_hu_tui_1.index:                                                   # 相互推荐的组合    
    list_hutui_1.append([data_hu_tui_1.loc[i][0],data_hu_tui_1.loc[i][1]]) # [userid,t_userid]

list_hutui_1 = list(set([tuple(set(i)) for i in list_hutui_1]))        
list_hutui_2 = []
for j in list_hutui_1:                                   # 找到相互推荐数据（成对出现）的一条，确定在下线出现多次的相互推荐数据
    if len(j)==1:
        list_hutui_2.append(data1[(data1.userid==j[0])&(data1.t_userid==j[0])])     # 自身推荐自身的节点
    else:
        if data1.userid.tolist().count(j[0])>=2:             # A=j[0],B=j[1] 有 A 在下线的数量大于等于2则 A 有除 B 以外的其他上线 删除 B 推荐 A
            list_hutui_2.append(data1[(data1.userid==j[0])&(data1.t_userid==j[1])])
        else:
            list_hutui_2.append(data1[(data1.userid==j[1])&(data1.t_userid==j[0])])
if list_hutui_2 != []:
    list_hutui_3 = pd.concat(list_hutui_2,sort=False).index
    data1.drop(index=list_hutui_3,inplace=True)


# In[294]:


data_hu_tui_1


# ## 2数据节点

# In[256]:


### 所有节点数据
# 列表形式节点
# 列表形式节点
jiedian_data = list(set(list(chain(*data1.values))))
print(jiedian_data[0:5])
print("节点长度：",len(jiedian_data))


# ### 有推荐的节点和无推荐的节点分离
# #### 无推荐的节点不进行计算，缩小数据量

# In[260]:


node_df = pd.merge(data1,data1,how="left",left_on="userid",right_on="t_userid",suffixes=("","_y"))
node_df.head()


# In[261]:


# 没有推荐下线数据 ：userid 不在 t_userid 中的数据
node_1 = node_df[node_df.userid_y.isnull()]
list_node_1 = np.unique(node_1.userid.tolist())
list_node_1


# In[262]:


# 有推荐下线的节点
list_node_2 = np.unique(node_df.t_userid.tolist())


# In[263]:


len(list_node_2)


# In[265]:


data1.head()


# In[266]:


# 有推荐的节点与调整数据整合
data_tuijian = pd.DataFrame({"node":list_node_2})
j = -1
i = 0
while j != 0:   
    i +=1
    data_tuijian = pd.merge(data_tuijian,data1,how="left",
                            left_on=data_tuijian.columns[-1],
                            right_on=data1.t_userid,suffixes=("","_%s" % i),)
    data_tuijian.drop(columns=["t_userid"],inplace = True)
    
    j = len(data_tuijian[data_tuijian.columns[-1]].value_counts())
    #print("最大层级":i)


# In[268]:


get_ipython().run_cell_magic('time', '', '##  统计发展下线深度和人数\n\n#fuzhu = []         # 辅助任务 1\n\nover = pd.DataFrame(columns=["node","下限","count"])\nfor name,group_i in data_tuijian.groupby("node"): \n    print(name)                          # 节点    name :节点 ； group ：与name对应的子树\n    group_j = group_i.dropna(axis=1,how="all")\n    xia = np.unique(list(chain(*group_j.values)))\n    xxx = list(filter(lambda x : x!="nan" ,xia))\n    c = len(xxx)-1                                # 数量\n    #print(c)\n    #print(j,y,c)     \n    over = over.append({"node":name,"下限":len(group_j.columns)-1,"count":c},ignore_index=True)\n    \n    # 小批量测试\n   # 测试用的 可删除\n      #  break')


# In[271]:


over = over.append(pd.DataFrame({"node":list_node_1}),ignore_index=True,sort=False)
over.fillna(0)
over.head()


# In[273]:


over.to_excel("计算结果.xlsx",index=False)
print(len(over))

