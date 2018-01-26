# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:50:30 2017
@author: zehao
"""

import numpy as np 
import pandas as pd 
import scipy as sp 
import matplotlib.pyplot as plt
import re
import os

AdmData = pd.read_csv('AdmData.csv')
AlarmTable = pd.read_csv('AlarmTable_v2.csv', error_bad_lines = False)
ComplicationTime = pd.read_csv('ComplicationTimes.csv', error_bad_lines = False)
DescriptionTable = pd.read_csv('DescriptionTable.csv', error_bad_lines = False, encoding = 'ISO-8859-1')

def AlarmMessage_indicator():
	''' extract the alarm message from the alarm message'''
	import re
	pattern = re.compile(r'([A-Z]+\d*\s*)+|\s') #pattern is used to match the type of alarm
	Alarm_type=[]
	n=0
	for row in AlarmTable.iterrows():
		if not isinstance(row[1]['AlarmMessage'],str):
			continue
		else:
			matc = pattern.match(row[1]['AlarmMessage'])
			Alarm_type.append(matc.group())
		Alarm_type_levels = list(set(Alarm_type))
	level_label = zip(range(len(Alarm_type_levels)),Alarm_type_levels)
	return {n:alarm_type for n,alarm_type in level_label}


def search_correspond(dic,val):
	''' to find the indicator of the pattern in the dic'''
	Alarm_type_dict = AlarmMessage_indicator()
	for k in dic:
		if dic[k]==val:
			k_corresponding = k
			break
	return k_corresponding


def data_filter(Data_all,Lag):
	'''filter of the data'''
	data = [Data_all[1],Data_all[2],Data_all[3],Data_all[4]]
	min_start_time = min(Data_all[5])
	max_start_time = max(Data_all[5])
	selected_data = [[],[],[],[]]
	for t in range(len(Data_all[0])):
		start_end_time = zip(Data_all[5],Data_all[5]+Data_all[7])
		g_start_end_time = (se for se in start_end_time)
		for se_t in g_start_end_time:
			if se_t[0]-Lag<=p_2222_partime[t] and p_2222_partime[t]<=se_t[1]-Lag:
				for i in range(4):
					selected_data[i].append(data[i][t])
				break
	return delected_data()

class Patient(object):
	'''Patiet object is used to store al relevant information of a certain paient'''
	def __init__(self,id):
		self.id = id

	def patient_alarm_record(self):
		store_value_list=[[],[],[],[],[],[],[],[]]
		for row in AlarmTable.iterrows():
			if((row[1]['ID']==self.id)):
				ind = 0
				for val in row[1]:
					store_value_list[ind].append(val)
					ind += 1
		col_name = ['ID','alarmstarttime','AlarmLevel','AlarmParcode','AlarmMessage','AlarmDuration']
		store_value_list = store_value_list[1:7]
		Patient = pd.DataFrame({k:v for k,v in zip(col_name,store_value_list)})
		return Patient 

	def patient_certain_alarm_message(self,k):
		# k is the key in dictionary corresponding to alarm type of interest
		Patient = self.patient_alarm_record()
		store_value_list=[[],[],[],[],[],[]]
		pattern = re.compile(Alarm_type_dict[int(k)])
		for row in Patient.iterrows():
			if isinstance(row[1]['AlarmMessage'],str) and pattern.match(row[1]['AlarmMessage']):
				ind = 0
				for val in row[1]:
					store_value_list[ind].append(val)
					ind += 1
		col_name = ['AlarmDuration','AlarmLevel','AlarmMessage','AlarmParcode','ID','alarmstarttime']
		Patient_alarm_message = pd.DataFrame({k:v for k,v in zip(col_name,store_value_list)})
		patient_alarm_message = Patient_alarm_message['AlarmMessage'].as_matrix()
		levels_of_message = [int(n.split()[-1]) for n in patient_alarm_message]
		start_time = Patient_alarm_message['alarmstarttime'].as_matrix()
		level = Patient_alarm_message['AlarmLevel'].as_matrix()
		parcode = Patient_alarm_message['AlarmParcode'].as_matrix()
		duration = Patient_alarm_message['AlarmDuration'].as_matrix()
		return levels_of_message, start_time, level, parcode, duration

	def patient_physiological_data(self):
		if not os.path.exists('/Users/zehaodong/Desktop/alarm_fatigue/data/NICU_Physiological_Data/%d.csv'%(self.id)):
			print('There are no relevant physiological data for patient with id %d'%(self.id))
		else:
			df_phy_data = pd.read_csv('/Users/zehaodong/Desktop/alarm_fatigue/data/NICU_Physiological_Data/%d.csv'%(self.id))
		partime = df_phy_data['ParTime'].as_matrix()
		low_bp = df_phy_data['AR1.D'].as_matrix()
		mean_bp = df_phy_data['AR1.M'].as_matrix()
		high_bp = df_phy_data['AR1.S'].as_matrix()
		HR = df_phy_data['HR'].as_matrix()
		cvp_2 = df_phy_data['CVP2'].as_matrix()
		return partime, low_bp, mean_bp, high_bp,HR,cvp_2

	def patient_whole_alarm_info(self):
		Patient = self.patient_alarm_record()
		start_time = Patient['alarmstarttime'].as_matrix()
		level = Patient['AlarmLevel'].as_matrix()
		parcode = Patient['AlarmParcode'].as_matrix()
		duration = Patient['AlarmDuration'].as_matrix()
		return start_time, level, parcode, duration    

if __name__ == '__main__':
	import numpy as np 
	import pandas as pd 
	import scipy as sp 
	import matplotlib.pyplot as plt
	import re
	import os
