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


1.直接进行values维度计算利用timeit监测两种方式的运行时间%%timeit

df['amount'] = df['last_kj_amount'] + df['last_15day_amount']567 µs ± 32 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)%%timeit

df['amount2'] = df['last_kj_amount'].values + df['last_15day_amount'].values221 µs ± 19 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)在当前数据量并不大的情况下，速度只提升了一倍，如果在生产中使用values值进行计算的话优化效果更好。2.利用np.vectorize代替apply这里我们对上面计算的结果进行向上取整，采用两种方式对比，并用timeit监测运行时间。%%timeit
df['safe_amount1'] = df.apply(lambda x:math.ceil(x['amount']),axis = 1)243 ms ± 16.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)def getInt(x):
    return math.ceil(x)

%%timeit
df['safe_amount2'] = np.vectorize(getInt)(df['amount'])4.05 ms ± 162 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)可见采用np.vectorize方式，时间缩短了近60倍。3.利用np.where代替applynp.where类似于if…else语句，主要是用来进行条件判定和数据筛选。%%timeit

df['signal'] = df.apply(lambda x: 1 if x['last_15day_amount'] == 0 and x['statis_date'] == 20200506 and x['time_type'] == 2 else 0,axis = 1)269 ms ± 31.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)%%timeit

df['signal1'] = np.where((df['last_15day_amount'] == 0)&(df['statis_date'] == 20200506)&(df['time_type'] == 2),1,0)1.59 ms ± 196 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)可见，np.where方式时间缩短了近170倍。

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
                                
                                
# sklearn trick

https://github.com/yiyuezhuo/pyro-tutorial-ch/blob/master/intro_part_i.ipynb


# matplotlib pyplot trick

ax = df.plot()
ax.set_xlabel('BTD_WV-IR11(K)', fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import axisartist

fig = plt.figure()
ax = axisartist.Subplot(fig, 111)
fig.add_subplot(ax)
ax.plot(df)
# 横坐标标签中`WV-IR11`为下标
ax.set_xlabel(r'$BTD_{WV-IR11}$(K)')  # fontdict和fontsize不起作用，要通过下一行调整字号
ax.axis['bottom'].label.set_size(20)
# 图例显示出中文
font = FontProperties(size=15, fname=r'C:\Windows\Fonts\simsun.ttc')
ax.legend(['准确率', '识别率', '调和均值'], prop=font, frameon=False)  # 指定prop后fontsize参数失效
# 纵坐标刻度用百分比表示
def to_percent(temp, position):
    return f'{100*temp:.2f}%'
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
# 绘制x=-2的虚线
ax.axvline(-2, color='red', linestyle ='--')
# 绘制坐标轴箭头
ax.axis['bottom'].set_axisline_style('->')
ax.axis['left'].set_axisline_style('->')
# 不显示上边框和右边框
ax.axis['top'].set_visible(False)
ax.axis['right'].set_visible(False)

# pandas plot with ax
fig = plt.figure("fig6", constrained_layout=True, dpi=200)
ax = plt.axes()
style = [f"k{m}-" for m in ['', 'o', 's', '^']]
df.plot(ax=ax, style=style)
ax.set_xlabel("Time")
ax.set_ylabel("Y")
ax.set_xticks(df.index)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.spines["top"].set_color("none")  # 不显示上框线
ax.spines["right"].set_color("none")  # 不显示右框线

# matplot lib add text and annotations
ser = np.random.randn(100) #time series data
fig = plt.figure(figsize=(12,8)) 
ax = fig.add_subplot(111)
plt.grid()
ax.plot(ser)
#title
ax.set_title('Time Series Plot', fontsize=18, fontweight='bold')
#axis title
ax.set_xlabel('Day', fontsize=15)
ax.set_ylabel('Value', fontsize=15)
#axis limits
ax.axis([0,100, -5, 5])
#text
ax.text(3, 4, 'Random Noise', style='italic', fontsize=12,
        bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
ax.text(60, 4.1, 'No continuous trend observed', fontsize=15,  color='red')
#annotation
ax.plot([33.5], [2.5], 'o')
ax.annotate('peak', xy=(33.5, 2.5), xytext=(40, 4), fontsize=12,
            arrowprops=dict(facecolor='black', shrink=0.05))

