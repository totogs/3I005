import re
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

A=['A','C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_']

Dtrain=[]#Matrice des acides aminées (L'alignement

#Lecture du fichier Dtrain.txt et conversion des séquences en matrice
with open("PF00018/Dtrain.txt", "r") as f_seq:

	for line in f_seq.readlines():
	
		if(line[0]!='>'):
			sequence=[]
			
			for char in line:
				if(char!='\n'):
					sequence.append(char)
					
			Dtrain.append(sequence)
			
Dtrain=np.array(Dtrain)
print(Dtrain.shape)

M=Dtrain.shape[0]#M est le nombre de séquences
L=Dtrain.shape[1]#L est la longueur d'une séquence





#Premiere fonction

def occurences(data):

	dict_occ = dict()
	
	
	for i in range(L):
	
		d = dict()
		for m in range(M):
		
			if(data[m,i] in d):
				d[data[m,i]]+=1
			else:
				d[data[m,i]]=1
		
		dict_occ[i]=d	
				
	return dict_occ
	



		
	
Occ = occurences(Dtrain)
#Verification que la somme des occurences pour tout a sur chaque colonne i est égal à M
"""
for i in Occ.values():

	cpt=0
	
	for val in i.values():
	
		cpt+=val
		
	print(cpt)
	
"""






