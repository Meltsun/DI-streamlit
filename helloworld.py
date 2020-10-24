import numpy as np
import pandas as pd
import streamlit as st
import random

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False 
import matplotlib.pyplot as plt

@st.cache
def load_dataset():
	return pd.read_csv("originSet.csv",header=None,index_col=None ).values

@st.cache
def load_features():
	return pd.read_csv("feature.csv",index_col=None,header=0,encoding='gbk')

#定义通用的变量
dataSet=load_dataset()#所有数据
feature=load_features()#所有特征值
dsTypeN = ['无','车辆经过','浇水','敲击','攀爬']
dsCruveN = ['时域','时域差分','小波包分解','经验模态分解']
featureN=feature.columns#特征值名


st.title('分布式光纤传感系统的扰动信号分析')
st.title('【本程序仍在开发中，预计11月初完成】')
agree = st.sidebar.radio("模式",['特征值分布分析','单个样本分析','分类器测试'])
st.sidebar.write('---')

if(agree=='单个样本分析'):
	#获取侧栏的控制字
	dsType = st.sidebar.selectbox('扰动类型' , dsTypeN)
	
	dsNumber = st.sidebar.slider('样本编号',1,150,50,1)-1#整数，0-149
	st.sidebar.write('---')
	dsCruve = st.sidebar.selectbox('图像类型' , dsCruveN)#字符串
	
	#小标题
	st.title(f'【{dsType}扰动信号No.{dsNumber}】- {dsCruve} ')

	dsType = dsTypeN.index(dsType)#整数，0-4
	#单个样本分析中的通用变量
	data=dataSet[dsType*3+dsNumber//50][dsNumber%50*22:dsNumber%50*22+33]
	
	#dFeature
	if(dsCruve=='时域'):
		#ser=pd.Series(data.tolist())
		#ser.plot()
		st.line_chart(data)
		#st.pyplot()
		#图
		dFeature=feature.iloc[dsType*150+dsNumber][:-1][0:15]
		st.dataframe(pd.DataFrame(dFeature[0:8]).T)
		st.dataframe(pd.DataFrame(dFeature[8:]).T)

	elif(dsCruve=='时域差分'):
		dataDiff=np.diff(data)
		#图
		figure1=plt.figure()
		plt.plot([i for i in range(0,len(dataDiff))],dataDiff)
		st.pyplot(figure1)

		dFeature=feature.iloc[dsType*150+dsNumber][:-1][15:30]
		dFeature.index=[i.replace('差分-','') for i in dFeature.index]
		st.dataframe(pd.DataFrame(dFeature[0:8]).T)
		st.dataframe(pd.DataFrame(dFeature[8:]).T)

	elif(dsCruve=='小波包分解'):
		dFeature=feature.iloc[dsType*150+dsNumber][:-1][30:40]
		st.dataframe(pd.DataFrame(dFeature[0:5]).T)
		st.dataframe(pd.DataFrame(dFeature[5:]).T)
		dsCruveWP = st.sidebar.selectbox('选择小波图像' , ['最低频','低频','高频','最高频'])
		#根据dsCruveWP的值（字符串）画不同的图

	else:
		'（还在做 -_-||）'

elif(agree=='特征值分布分析'):
	#获取各种控制字
	featureX = st.sidebar.selectbox('X' , featureN[:-1])#字符串
	featureY = st.sidebar.selectbox('Y' , [i for i in featureN if i !=featureX] ,index=39 )#字符串
	st.sidebar.text('* Y设置为 lable 可查看一维特征分布')
	st.sidebar.write('---')
	dsShow = [st.sidebar.checkbox(dsTypeN[i]) for i in range(0,5)]
	st.sidebar.write('---')
	dsShow = [i for i in range(0,5) if dsShow[i]]#获得被所有勾选的类型编号组成的列表
	#根据选择的类别和特征值画散点图:
	colorN=['red','blue','green','black','yellow']
	markerN=['x','o','*','v','s']
	figure2=plt.figure()
	X = feature[featureX]
	Y = feature[featureY]
	plt.xlabel(featureX)
	plt.ylabel(featureY)
	for i in dsShow:
		XX = X.iloc[150*i:150*i+150]
		YY = Y.iloc[150*i:150*i+150]
		plt.scatter(XX,YY,marker = markerN[i],color = colorN[i], s = 40 ,label = dsTypeN [i])
	if(not st.sidebar.checkbox("关闭图例")):
		plt.legend(loc = 'best')
	st.pyplot(figure2)

elif(agree=='分类器测试'):
	st.sidebar.write('超参数设置')
	knnPointNum = st.sidebar.number_input('KNN-点数', min_value=1, max_value=20, value=5, step=1)
	if(st.sidebar.button('启动分类器测试')):
		dataID=[i for i in range(0,150)]
		trainNum=120
		trainID=np.empty([5,trainNum],dtype=int)
		testID=np.empty([5,150-trainNum],dtype=int)
		for i in range(0,5):
			j=random.sample(dataID, trainNum)
			trainID[i]=np.array(j)+i*150
			testID[i]=np.array([i for i in dataID if i not in j])+i*150
		trainFeature=feature.iloc[trainID.ravel()]
		testFeature=feature.iloc[testID.ravel()]



else:
	st.write("没做完 = =")

st.write('---')
st.write('请在侧栏从上至下依次设置参数')
st.text('点击左上角 > 打开侧栏')
st.text('发生错误时请尝试点击右上角 ☰ → Clear cache 清除缓存')
st.text('本应用基于python-streamlit ， 右上角有小人在动即正在加载，请稍等')
st.text('数据更新时间：'+'2020年10月19日00:36:39')
st.text('by BJTU_WXY')