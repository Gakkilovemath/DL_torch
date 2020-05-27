# DL_torch
 
http://tangshusen.me/Dive-into-DL-PyTorch/#/

# pandas show tricks

#Using hide_index() from the style function

df.head(10).style.hide_index()

#Using hide_columns to hide the unnecesary columns

df.head(10).style.hide_index().hide_columns(['method','year'])

#Highlight the maximum number for each column

df.head(10).style.highlight_max(color = 'yellow')

df.head(10).style.highlight_min(color = 'lightblue')

#Adding Axis = 1 to change the direction from column to row

planets.head(10).style.highlight_max(color = 'yellow', axis =1)

#Higlight the null value

planets.head(10).style.highlight_null(null_color = 'red')

# seabron tricks

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font=’Franklin Gothic Book’,
        rc={
 ‘axes.axisbelow’: False,
 ‘axes.edgecolor’: ‘lightgrey’,
 ‘axes.facecolor’: ‘None’,
 ‘axes.grid’: False,
 ‘axes.labelcolor’: ‘dimgrey’,
 ‘axes.spines.right’: False,
 ‘axes.spines.top’: False,
 ‘figure.facecolor’: ‘white’,
 ‘lines.solid_capstyle’: ‘round’,
 ‘patch.edgecolor’: ‘w’,
 ‘patch.force_edgecolor’: True,
 ‘text.color’: ‘dimgrey’,
 ‘xtick.bottom’: False,
 ‘xtick.color’: ‘dimgrey’,
 ‘xtick.direction’: ‘out’,
 ‘xtick.top’: False,
 ‘ytick.color’: ‘dimgrey’,
 ‘ytick.direction’: ‘out’,
 ‘ytick.left’: False,
 ‘ytick.right’: False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})
                                
# TORCH TRICK
optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd) # 对权重参数衰减
optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减
optimizer_w.zero_grad()
optimizer_b.zero_grad()
l.backward()
对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
optimizer_w.step()
optimizer_b.step()
                                
                                
