import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection

matplotlib.use('Agg')

from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse',False)
#Set title

st.title('Data Analysis On Various Datasets')
image = Image.open('logo.jpeg')
st.image(image,use_column_width=True)


def main():
	st.title('Data Analysis On Various Datasets')
	activities=['EDA','visualisation','model','About us']
	option=st.sidebar.selectbox('Select option',activities)

	if option=="EDA":
		st.subheader("Exploratory Data Analysis")

		data=st.file_uploader("Upload datasets",type=['csv','xlsx','txt','html','json'])
		
		if data is not None:
			st.success("success")
			df=pd.read_csv(data)
			st.dataframe(df.head(50))
			
			if st.checkbox("Display shape"):
				st.write(df.shape)

			if st.checkbox("Display columns"):
				all_columns=df.columns.to_list()
				st.write(all_columns)

			if st.checkbox('Select Multiple columns'):
				selected_columns=st.multiselect("Select Preferred columns",df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

			if st.checkbox("Display summary"):
				st.write(df.describe().T)

			if st.checkbox("Null Values"):
				st.write(df.isnull().sum())
			if st.checkbox('Data Types'):
				st.write(df.dtypes)
			if st.checkbox('Display Correlation'):
				st.write(df.corr())
			

	elif option=="visualisation":
		st.subheader("Data visualisation")
		data=st.file_uploader("Upload datasets",type=['csv','xlsx','txt','html','json'])
		
		if data is not None:
			st.success("success")
			df=pd.read_csv(data)
			st.dataframe(df.head())
			if st.checkbox('Select Multiple Columns to plot'):
				selected_columns=st.multiselect("Select Preferred columns",df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			if st.checkbox("Display Heatmap"):
				st.write(sns.heatmap(df1.corr(), vmax=1, square=True, annot=True,cmap='viridis'))
				st.pyplot()
			if st.checkbox("Display Pairplot"):
				st.write(sns.pairplot(df1,diag_kind='kde'))
				st.pyplot()

			if st.checkbox("Pie Chart"):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox("Select a column, NB: Select Target column",all_columns)
				pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pieChart)
				st.pyplot()


# #Plotting multiple plots at once

			cols=df.columns.to_list()

			plots=st.selectbox("select a choice of plot",['histogram','bargraph','area plot','line plot'])
			selected_cols=st.multiselect("Select Your Preferred columns",df.columns)


			if st.button("Create Plot"):
				
				if plots=="area plot":
					df2=df[selected_cols]
					st.area_chart(df2)
					# st.success("success")
					st.pyplot()

				elif plots=="histogram":
					  df2=df[selected_cols]
					  st.dataframe(df2)
					  st.write(plt.hist(df2, bins=20))
					  st.success("success")
					  st.pyplot()


				elif plots=="bargraph":
					df2=df[selected_cols]
					st.dataframe(df2)
					st.bar_chart(df2)
					st.success("success")
					st.pyplot()

				elif plots=="line plot":
					df2=df[selected_cols]
					st.line_chart(df2)
					st.success("success")
					st.pyplot()


			


	elif option=="model":
		st.subheader("Model Building")

		data=st.file_uploader("Upload datasets",type=['csv','xlsx','txt','html','json'])
		st.success("success")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head()) 

			
			if st.checkbox('Select Multiple columns'):
				new_data=st.multiselect("Select column. NB: Make Sure Target column is selected last",df.columns)
				df1=df[new_data]
				st.dataframe(df1)
				X=df1.iloc[:,0:-1]
				y=df1.iloc[:,-1]


			seed = st.sidebar.slider('Seed', 1, 200)

			classifier_name = st.sidebar.selectbox('Select classifier',('KNN', 'SVM','LR','naive_bayes','DecisionTree'))

			 
			def add_parameter(name_of_clf):
			    params = dict()
			    if name_of_clf == 'SVM':
			        C = st.sidebar.slider('C', 0.01, 15.0)
			        params['C'] = C
			    else:
			        name_of_clf == 'KNN'
			        K = st.sidebar.slider('K', 1, 15)
			        params['K'] = K
			    return params

			#calling our function
			params = add_parameter(classifier_name)


			#accessing our classifier

			def get_classifier(name_of_clf, params):
			    clf = None
			    if name_of_clf == 'SVM':
			        clf = SVC(C=params['C'])
			    elif name_of_clf == 'KNN':
			        clf = KNeighborsClassifier(n_neighbors=params['K'])
			    elif name_of_clf=='LR':
			     	clf=LogisticRegression()
			    elif name_of_clf=='naive_bayes':
			     	clf=GaussianNB()
			    elif name_of_clf=='DecisionTree':
			    	clf=DecisionTreeClassifier()

			    else:
			        st.warning('select your choice of algorithm')
			    return clf


			clf = get_classifier(classifier_name, params)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			st.write('Predictions',y_pred)

			accuracy = accuracy_score(y_test, y_pred)

			st.write('Classifier name:',classifier_name)
			st.write('Accuracy:', accuracy)

	elif option=="About us":
		st.subheader("About us")
		st.write("Voila!!! We have successfully created our ML App")
		st.write("This is our ML App to make things eaier for our users to understand their data without a stress")


		st.balloons()



if __name__ =='__main__':
	main()