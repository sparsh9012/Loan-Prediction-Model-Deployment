from flask import Flask, request, render_template, jsonify, make_response, request
import pandas as pd
import os
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		df = pd.read_csv(request.files.get('file'))
		
		cols_afterdrop90more = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
       'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
       'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
       'pymnt_plan', 'desc', 'purpose', 'title', 'zip_code',
       'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
       'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
       'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
       'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
       'last_credit_pull_d', 'collections_12_mths_ex_med', 'policy_code',
       'application_type', 'acc_now_delinq', 'chargeoff_within_12_mths',
       'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens', 'hardship_flag',
       'disbursement_method', 'debt_settlement_flag']

		cols_afterdropuniquecat = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
       'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
       'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
       'desc', 'purpose', 'title', 'zip_code', 'addr_state',
       'dti', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths',
       'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
       'last_credit_pull_d', 'acc_now_delinq', 'delinq_amnt',
       'pub_rec_bankruptcies', 'tax_liens', 'debt_settlement_flag']
	   
		df = df.sample(frac=0.05)
		if 'loan_status' in df:
			df = df.drop('loan_status', axis=1)
		df = df[cols_afterdropuniquecat]
		df = df.drop(['emp_title','title','desc'], axis=1)

		#Label-encoding
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		df['sub_grade']=le.fit_transform(df['sub_grade'])
		df['addr_state']=le.fit_transform(df['addr_state'])

		df['zip_code'] = df['zip_code'].str.replace('xx', '').astype('int')
		df['zip_code']=le.fit_transform(df['zip_code'])

		#one-hot encoding
		df = pd.concat([df, pd.get_dummies(df[['term','grade','home_ownership','verification_status','purpose','debt_settlement_flag']])], axis=1)
		df = df.drop(['term','grade','home_ownership','verification_status','purpose','debt_settlement_flag'], axis=1)

		# emp-length encoding
		look_up = {'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,
				   '9 years':9,'10 years':10,'10+ years':11,'< 1 year':0}
		df['emp_length'] = df['emp_length'].apply(lambda x: look_up[x] if(str(x) != 'nan') else x)

		dt_cols = ['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
		df['issue_d_month'], df['issue_d_year'] = df['issue_d'].str.split('-', 1).str
		df['earliest_cr_line_month'], df['earliest_cr_line_year'] = df['earliest_cr_line'].str.split('-', 1).str
		df['last_pymnt_d_month'], df['last_pymnt_d_year'] = df['last_pymnt_d'].str.split('-', 1).str
		df['last_credit_pull_d_month'], df['last_credit_pull_d_year'] = df['last_credit_pull_d'].str.split('-', 1).str
		df = df.drop(dt_cols, axis=1)
		dt_cols_new = ['issue_d_month','issue_d_year','earliest_cr_line_month','earliest_cr_line_year','last_pymnt_d_month',
					   'last_pymnt_d_year','last_credit_pull_d_month','last_credit_pull_d_year']
		look_up = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
		df['issue_d_month'] = df['issue_d_month'].apply(lambda x: look_up[x] if(str(x) != 'nan') else x)
		df['earliest_cr_line_month'] = df['earliest_cr_line_month'].apply(lambda x: look_up[x] if(str(x) != 'nan') else x)
		df['last_pymnt_d_month'] = df['last_pymnt_d_month'].apply(lambda x: look_up[x] if(str(x) != 'nan') else x)
		df['last_credit_pull_d_month'] = df['last_credit_pull_d_month'].apply(lambda x: look_up[x] if(str(x) != 'nan') else x)
		df[dt_cols_new] = df[dt_cols_new].astype(float)
		df['int_rate'] = df['int_rate'].str.replace('%', '').astype('float')
		df['revol_util'] = df['revol_util'].str.replace('%', '').astype('float')
		df = df.rename(index=str, columns={'int_rate':'int_rate%', 'revol_util':'revol_util%'})

		#impute the missing values
		from sklearn.preprocessing import Imputer
		imputer = Imputer()
		df['tax_liens']=imputer.fit_transform(df[['tax_liens']].values)
		df['inq_last_6mths']=imputer.fit_transform(df[['inq_last_6mths']].values)
		df['acc_now_delinq']=imputer.fit_transform(df[['acc_now_delinq']].values)
		df['pub_rec']=imputer.fit_transform(df[['pub_rec']].values)
		df['open_acc']=imputer.fit_transform(df[['open_acc']].values)
		df['annual_inc']=imputer.fit_transform(df[['annual_inc']].values)
		df['delinq_2yrs']=imputer.fit_transform(df[['delinq_2yrs']].values)
		df['total_acc']=imputer.fit_transform(df[['total_acc']].values)
		df['delinq_amnt']=imputer.fit_transform(df[['delinq_amnt']].values)
		df['mths_since_last_delinq']=imputer.fit_transform(df[['mths_since_last_delinq']].values)
		df['pub_rec_bankruptcies']=imputer.fit_transform(df[['pub_rec_bankruptcies']].values)
		df['revol_util%']=imputer.fit_transform(df[['revol_util%']].values)
		df['emp_length']=imputer.fit_transform(df[['emp_length']].values)

		df['last_pymnt_d_year']=imputer.fit_transform(df[['last_pymnt_d_year']].values)
		df['last_pymnt_d_month']=imputer.fit_transform(df[['last_pymnt_d_month']].values)
		df['earliest_cr_line_year']=imputer.fit_transform(df[['earliest_cr_line_year']].values)
		df['earliest_cr_line_month']=imputer.fit_transform(df[['earliest_cr_line_month']].values)
		df['last_credit_pull_d_month']=imputer.fit_transform(df[['last_credit_pull_d_month']].values)
		df['last_credit_pull_d_year']=imputer.fit_transform(df[['last_credit_pull_d_year']].values)


		from sklearn.externals import joblib
		loaded_model = joblib.load('clf_model.pkl')
		predictions = loaded_model.predict(df)
		
		mydict = {1: 'Fully Paid', 2: 'Charged Off', 
		3: 'Does not meet the credit policy. Status:Fully Paid',
		4: 'Does not meet the credit policy. Status:Charged Off'}
		predictions_mapped = [mydict.get(n, n) for n in predictions]
		
		serial = np.arange(1,len(predictions_mapped)+1,1)
		result = dict(zip(serial, predictions_mapped))
		
		Table = []
		for key, value in result.items():
			temp = []
			temp.extend([key,value])  #Note that this will change depending on the structure of your dictionary
			Table.append(temp)
		Table
		
		return render_template('upload.html', table=Table)
	return render_template('upload.html')

if __name__ == '__main__':
	app.run(port=5000, debug=True)
