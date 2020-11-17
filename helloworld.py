import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from PIL import Image
import pywt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['font.serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False 
import matplotlib.pyplot as plt
from re import split
from Frequency_Feature import frequency_draw
from Time_Feature import  time_draw


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
	

def check_input(text):
	text=split('[，；;,\n\t]',text)
	text=[i for i in text if i!='' ]
	b=['1','2','3','4','5','6','7','8','9','0','.','*','/','+','-',' ']
	if(len(text)==0):
		return None,0
	for i in text:
		for j in i:
			if(j not in b):
				return None,0
		for j in b[:10]:
			if(j in i):
				break
		else:
			return None,0
	text=[eval(i) for i in text]	
	return text,len(text)

#定义通用的变量
dataSet=load_dataset()#所有数据
feature=load_features()#所有特征值
dsTypeN = ['无','车辆经过','浇水','敲击','攀爬']
dsCruveN = ['时域','时域差分','小波包分解','经验模态分解']
featureN=feature.columns#特征值名

st.title('超长周界安防预警系统')
st.title('测试与成果展示平台')
agree = st.sidebar.radio("模式",["简介",'单个样本分析','特征值分布分析','识别测试'])

if(agree=='单个样本分析'):
	st.sidebar.write('查看一个具体信号样本的各个波形图和相关的特征值')
	st.sidebar.write('---')
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
		#小波包分解
		wp=pywt.WaveletPacket(data=data,wavelet='db3',mode='symmetric',maxlevel=2) #2层小波包分解
		new_wp = pywt.WaveletPacket(data=None, wavelet='db3', mode='symmetric',maxlevel=2) #新建小波包树用来重构4个节点系数

		new_wp['aa'] = wp['aa']
		LL=new_wp.reconstruct(update=False)


		del(new_wp['aa'])
		new_wp['ad'] = wp['ad']
		LH=new_wp.reconstruct(update=False)
		

		del(new_wp['a'])
		new_wp['da'] = wp['da']
		HL=new_wp.reconstruct(update=False)


		del(new_wp['da'])
		new_wp['dd'] = wp['dd']
		HH=new_wp.reconstruct(update=False)

		dFeature=feature.iloc[dsType*150+dsNumber][:-1][30:40]
		dsCruveWP = st.sidebar.selectbox('选择小波图像' , ['最低频','低频','高频','最高频'])
		#根据dsCruveWP的值（字符串）画不同的图
		if(dsCruveWP=='最低频'):
			figure1=plt.figure()
			plt.plot([i for i in range(0,len(LL))],LL)
			plt.xlabel('样本点')
			plt.ylabel('分解系数')
			plt.title(dsCruveWP)
			st.pyplot(figure1)
		elif(dsCruveWP=='低频'):
			figure1=plt.figure()
			plt.plot([i for i in range(0,len(LH))],LH)
			plt.xlabel('样本点')
			plt.ylabel('分解系数')
			plt.title(dsCruveWP)
			st.pyplot(figure1)
		elif(dsCruveWP=='高频'):
			figure1=plt.figure()
			plt.plot([i for i in range(0,len(HL))],HL)
			plt.xlabel('样本点')
			plt.ylabel('分解系数')
			plt.title(dsCruveWP)
			st.pyplot(figure1)
		elif(dsCruveWP=='最高频'):
			figure1=plt.figure()
			plt.plot([i for i in range(0,len(HH))],HH)
			plt.xlabel('样本点')
			plt.ylabel('分解系数')
			plt.title(dsCruveWP)
			st.pyplot(figure1)
		st.dataframe(pd.DataFrame(dFeature[0:5]).T)
		st.dataframe(pd.DataFrame(dFeature[5:]).T)
	else:
		'（还在做 -_-||）'

elif(agree=='特征值分布分析'):
	#获取各种控制字
	st.sidebar.write("选定几种信号，以及2个特征值。查看这些特征值的二维分布情况。这能直观的反映出特征值对信号的区分作用。")
	st.sidebar.text('* Y设置为 lable 可查看一维特征分布')
	st.sidebar.write('---')
	dsShow = [st.sidebar.checkbox(dsTypeN[i]) for i in range(0,5)]
	st.sidebar.write('---')
	dsShow = [i for i in range(0,5) if dsShow[i]]#获得被所有勾选的类型编号组成的列表
	featureX = st.sidebar.selectbox('X' , featureN[:-1])#字符串
	featureY = st.sidebar.selectbox('Y' , [i for i in featureN if i !=featureX] ,index=39 )#字符串
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

elif(agree=='识别测试'):
	st.sidebar.write('选择一个给定的样本或手动输入样本数据，提取其特征值并对其类型进行识别。')
	st.write("---")
	source=st.sidebar.radio("数据来源",['使用提供的数据','自行填入数据'])
	if(source=='使用提供的数据'):
		ready=True
		st.sidebar.write("---")
		#获取侧栏的控制字
		dsType = st.sidebar.selectbox('扰动类型' , dsTypeN)
		target = dsType 
		dsType = dsTypeN.index(dsType)
		dsNumber = load_testID()[st.sidebar.slider('样本编号',1,30,11,1)-1] #整数
		data=dataSet[dsType*3+dsNumber//50][dsNumber%50*22:dsNumber%50*22+33]
		dFeature=feature.iloc[dsType*150+dsNumber,:-1]
		st.write("样本数据如下：")
		st.dataframe(pd.DataFrame(data).T)
		st.write("样本的特征值如下：")
		st.dataframe(pd.DataFrame(dFeature).T)
		dFeature=dFeature.values
		result=infer(dFeature)
		st.write(f"两步分类-SVM分类：本样本对应的扰动类型为 {dsTypeN[result]}扰动 。")
		st.write(f"本样本的实际扰动类型为 {target}扰动。")
		if(dsType==result):
			st.write("推理正确 。")
		else:
			st.write(f"推理错误 。")

	else:
		"功能仍在完善中，尚未启用"
		text=st.text_area('请在此输入至少33个数据,用逗号或回车分割', value='', key=None)
		data,dataN=check_input(text)
		ready=False
		if(data== None):
			'请检查数据，确保数据和格式正确。'
		else:
			if(dataN>=33):
				'请点击按钮开始识别'
				ready=True
				data=np.array(data)
			else:
				f'数据数量不足({dataN})，请输入更多数据。'
		ready = ready and st.button('开始识别')
		if(ready):
			st.write("样本数据如下：")
			st.dataframe(pd.DataFrame(data).T)
			dFeature=np.empty(40)
			dFeature[:30] = time_draw(data)
			dFeature[30:] = frequency_draw(data)
			st.write("样本的特征值如下：")
			st.dataframe(pd.DataFrame(dFeature).T)
			result=infer(dFeature)
			st.write(f"两步分类-SVM分类：本样本对应的扰动类型为 {dsTypeN[result]}扰动 。")

	st.sidebar.write("---")
	dsCruve = st.sidebar.selectbox('查看信号波形和相关特征值' , ["(关闭)"]+dsCruveN)#字符串
	st.write("---")

	if(ready):
		if(dsCruve=='时域'):
			#ser=pd.Series(data.tolist())
			#ser.plot()
			st.line_chart(data)
			#st.pyplot()
			#图
			st.dataframe(pd.DataFrame(dFeature[0:8]).T)
			st.dataframe(pd.DataFrame(dFeature[8:]).T)

		elif(dsCruve=='时域差分'):
			st.line_chart(np.diff(data))
			st.dataframe(pd.DataFrame(dFeature[0:8]).T)
			st.dataframe(pd.DataFrame(dFeature[8:]).T)

		elif(dsCruve=='小波包分解'):
			#小波包分解
			wp=pywt.WaveletPacket(data=data,wavelet='db3',mode='symmetric',maxlevel=2) #2层小波包分解
			new_wp = pywt.WaveletPacket(data=None, wavelet='db3', mode='symmetric',maxlevel=2) #新建小波包树用来重构4个节点系数

			new_wp['aa'] = wp['aa']
			LL=new_wp.reconstruct(update=False)


			del(new_wp['aa'])
			new_wp['ad'] = wp['ad']
			LH=new_wp.reconstruct(update=False)
			

			del(new_wp['a'])
			new_wp['da'] = wp['da']
			HL=new_wp.reconstruct(update=False)


			del(new_wp['da'])
			new_wp['dd'] = wp['dd']
			HH=new_wp.reconstruct(update=False)

			dsCruveWP = st.sidebar.selectbox('选择小波图像' , ['最低频','低频','高频','最高频'])
			#根据dsCruveWP的值（字符串）画不同的图
			if(dsCruveWP=='最低频'):
				figure1=plt.figure()
				plt.plot([i for i in range(0,len(LL))],LL)
				plt.xlabel('样本点')
				plt.ylabel('分解系数')
				plt.title(dsCruveWP)
				st.pyplot(figure1)
			elif(dsCruveWP=='低频'):
				figure1=plt.figure()
				plt.plot([i for i in range(0,len(LH))],LH)
				plt.xlabel('样本点')
				plt.ylabel('分解系数')
				plt.title(dsCruveWP)
				st.pyplot(figure1)
			elif(dsCruveWP=='高频'):
				figure1=plt.figure()
				plt.plot([i for i in range(0,len(HL))],HL)
				plt.xlabel('样本点')
				plt.ylabel('分解系数')
				plt.title(dsCruveWP)
				st.pyplot(figure1)
			elif(dsCruveWP=='最高频'):
				figure1=plt.figure()
				plt.plot([i for i in range(0,len(HH))],HH)
				plt.xlabel('样本点')
				plt.ylabel('分解系数')
				plt.title(dsCruveWP)
				st.pyplot(figure1)
			st.dataframe(pd.DataFrame(dFeature[0:5]).T)
			st.dataframe(pd.DataFrame(dFeature[5:]).T)
	
elif(agree=="简介"):
	'---'
	st.header("项目概述")
	"""
	"超长周界安防预警系统"是一个用于国境线、输油管线、机场或大型基地周边等场景的长周界安防预警系统。
	本系统将与新型传感器紧密结合，使用多种机器学习算法和信号处理手段，最终实现对入侵行为的自动识别与报警
	。本项目即是为上述系统设计算法并开发软件。
	"""
	image = Image.open('image1.png')
	st.image(image, caption='分布式光纤扰动传感系统',width=690)

	st.header("项目进展")
	"""
	 ● 完整的信号处理和识别程序
	\n ● 提取了40维特征值，比较全面的刻画了信号特征
	\n ● 测试了SVM、GRNN、BP、KNN等机器学习算法，最终使用KNN和SVM结合的算法和一个小样本集训练了分类器"""
	st.markdown('** ● 针对8微秒的信号样本，正确率达到90%**')
		
	image = Image.open('image2.png')
	st.image(image, caption='核心算法的测试结果',width=500)

	st.header("未来计划")
	"""
	\n ● 目前程序实时性差，需要优化程序结构提高实时性
	\n ● “浇水”和“攀爬”信号误报相对较多，拉低了整体识别率；
	\n   将会针对性的优化信号处理和机器学习算法以改善识别率
	\n ● 完善人机交互界面
	"""

	st.header("关于本应用")
	"""本应用用于展示项目成果，以及以可视化的方式进行数据分析与相关测试。
	请在侧栏从上至下依次设置参数。具体请参考各个功能下的说明。""" 

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