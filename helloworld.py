import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from PIL import Image

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False 
import matplotlib.pyplot as plt

@st.cache
def load_classifier():
	return pickle.load(open("svmClassifiers.pkl",'rb'))

def load_classifier_2():
	return pickle.load(open("knnClassifier.pkl",'rb'))

@st.cache
def load_standardizer():
	return pickle.load(open("standardizer.pkl",'rb'))

@st.cache
def load_dataset():
	return pd.read_csv("originSet.csv",header=None,index_col=None ).values

@st.cache
def load_features():
	return pd.read_csv("feature.csv",index_col=None,header=0,encoding='gbk')

@st.cache
def load_testID():
	return pd.read_csv("testID.csv",header=None,index_col=None ).values.ravel()

def infer(thisFeature):
	standardizer = load_standardizer()
	thisFeature = standardizer.transform([thisFeature])
	svmClassifier=load_classifier()
	knnClassifier=load_classifier_2()
	#得到验证样本的概率
	knnResult=knnClassifier.predict_proba(thisFeature)
	st.write("---")
	st.write("两步分类-KNN粗分类；本样本为5种类型的概率分别为：")
	knnResult
	knnResult=knnResult[0]
	#由knnResult分析得到 knn无法确定的样本的2个最大可能性
	possibility=np.array([3,4])
	maxID=knnResult.argmax()
	maxP=knnResult[maxID]
	if(maxP==1):
		if(maxID!=4):
			possibility[0]=maxID
			possibility[1]=4
	else:
		possibility[0]=maxID
		maxP=0
		for j in range(0,5):
			if(j!=maxID and knnResult[j]>maxP):
				possibility[1]=j
		possibility.sort()

	x=svmClassifier[possibility[0]][possibility[1]].predict(thisFeature)
	return possibility[round(x[0])]
	

#定义通用的变量
dataSet=load_dataset()#所有数据
feature=load_features()#所有特征值
dsTypeN = ['无','车辆经过','浇水','敲击','攀爬']
dsCruveN = ['时域','时域差分','小波包分解','经验模态分解']
featureN=feature.columns#特征值名

st.title('超长周界安防预警系统')
st.title('测试与成果展示平台')
agree = st.sidebar.radio("模式",["简介",'单个样本分析','特征值分布分析','扰动识别测试'])
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
		st.line_chart(np.diff(data))
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
	st.write("查看指定种类信号的指定两维特征值的分布情况")
	st.write("汉字乱码问题待修复")
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

elif(agree=='扰动识别测试'):
	st.write("---")
	source=st.sidebar.radio("数据来源",['使用提供的数据','自行填入数据'])
	if(source=='使用提供的数据'):
		st.sidebar.write("---")
		#获取侧栏的控制字
		dsType = st.sidebar.selectbox('扰动类型' , dsTypeN)
		target = dsType 
		dsType = dsTypeN.index(dsType)
		dsNumber = load_testID()[st.sidebar.slider('样本编号',1,1,30,1)-1] #整数
		data=dataSet[dsType*3+dsNumber//50][dsNumber%50*22:dsNumber%50*22+33]
		dFeature=feature.iloc[dsType*150+dsNumber,:-1]
		st.write("样本数据如下：")
		st.dataframe(pd.DataFrame(data).T)
		st.write("样本的特征值如下：")
		st.dataframe(pd.DataFrame(dFeature).T)

		dFeature=dFeature.values
		
	else:
		"功能仍在完善中，尚未启用"
		text=st.text_area('请在此输入33个数据,用逗号或回车分割', value='', key=None)
		if(st.button('填入数据')):
			st.write("= = 还没做完")
		dFeature=[0 for i in range(0,40)]

	
	if(source=='使用提供的数据'):
		result=infer(dFeature)
		st.write(f"两步分类-SVM分类：本样本对应的扰动类型为 {dsTypeN[result]}扰动 。")
		st.write(f"本样本的实际扰动类型为 {target}扰动。")
		if(dsType==result):
			st.write("推理正确 。")
		else:
			st.write(f"推理错误 。")

	st.sidebar.write("---")
	dsCruve = st.sidebar.selectbox('查看信号波形和相关特征值' , ["(关闭)"]+dsCruveN)#字符串
	st.write("---")
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
		st.line_chart(np.diff(data))
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
	
elif(agree=="简介"):
	'---'
	st.header("项目概述")
	"""
	"超长周界安防预警系统"是一个用于国境线、输油管线、机场或大型基地周边等场景的长周界安防预警系统。
	本系统将与新型传感器紧密结合，使用多种机器学习算法和信号处理手段，最终实现对入侵行为的自动识别与报警
	。本项目即是为上述系统设计算法并开发软件。
	"""
	image = Image.open('image1.png')
	st.image(image, caption='分布式光纤扰动传感系统',width=600)

	st.header("项目进展")
	"""
	目前信号处理和机器学习的核心算法已经基本设计完成。针对8微秒的信号样本，我们提取了总共40维特征值，
	使用KNN+SVM结合的方法对其进行模式识别，正确率达到90%。算法和代码层面的优化仍在持续进行中。
	"""
	'客户端软件(即本项目的最终成果）将在核心算法优化完成后开始开发。在核心算法确定且机器学习分类器训练完成的情况下，预计该软件的开发周期不会太长。'
	
	image = Image.open('image2.png')
	st.image(image, caption='核心算法的测试结果',width=400)

	st.header("关于本应用")
	"""本应用用于展示项目的中期成果，以及以可视化的方式进行数据分析与相关测试。请在侧栏从上至下依次设置参数。功能介绍如下：
	\n单个样本分析：查看一个信号样本的波形图和相关的特征值查看样本。
	\n特征值分布分析：选定几种信号，查看这些信号的2种特征值的二维分布。这能直观的反映出特征值对不同信号的区分作用。
	\n扰动识别测试：手动输入一组样本或者选择一组给定的样本，并使用机器学习分类器对样本的扰动类型进行识别。
	""" 

else:
	st.write("没做完 = =")

if(agree!="简介"):
	st.write('---')
	st.write('请在侧栏从上至下依次设置参数')
	st.text('点击左上角 > 打开侧栏')
	st.text('发生错误时请尝试点击右上角 ☰ → Clear cache 清除缓存')
	st.text('本应用基于python-streamlit ， 右上角有小人在动即正在加载，请稍等')
	st.text('数据更新时间：'+'2020年11月13日18:38:19')
	st.text('by BJTU_WXY')