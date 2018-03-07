import re
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

A=['A','C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

Dtrain=[]#Matrice des acides aminees (L'alignement

#Lecture du fichier Dtrain.txt et conversion des sequences en matrice

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

M=Dtrain.shape[0]#M est le nombre de sequences
L=Dtrain.shape[1]#L est la longueur d'une sequence


B=[]

with open("PF00018/test_seq.txt", "r") as f_seq:

	for line in f_seq.readlines():
	
		if(line[0]!='>'):
			
			for char in line:
				if(char!='\n'):
					B.append(char)
					


B=np.array(B)
N=B.shape[0]
print(N)

#Premiere fonction

def occurences(data):

	dict_occ = dict()
	#on initialise les valeurs du dico a 0 pour eviter les ErroKey
	for i in range(L):
		d=dict()
		for a in A:
			
			d[a]=0
		dict_occ[i]=d
		
		
	
	for i in range(L):
	
		for m in range(M):
		
			d[data[m,i]]+=1
		
		dict_occ[i]=d	
				
	return dict_occ
	



		
	
Occ = occurences(Dtrain)
#Verification que la somme des occurences pour tout a sur chaque colonne i est egal a M
"""
for i in Occ.values():

	cpt=0
	
	for val in i.values():
	
		cpt+=val
		
	print(cpt)
	
"""
def poid(occurence, q):
	# a acide aminee
	# poids
	# M nombre total de sequences
	# L taille du dictionnaire
	# q tq les wi(a) forment la matrice de taille L*q ici 21
	# on prend en entree un dico
	
	#dictionnaire de dictionnaires w[colonne] = dictionnaire des poids des acides en position i
	w = dict()
	for i,liste in occurence.items():
		
		wi=dict()
		for prot,occ in liste.items():
			
			wi[prot]=float(occ+1)/(M+q)
			
		w[i]=wi
		
	
	return w

def entropie(wi, q):
	# Si = log2(q) + sum a dans A (wi(a).log2[wi(a)])
	# A tout les acides en position

	
	somme = math.log(q, 2);

	for a in wi.values():
	
		somme+=a*math.log(a, 2)

	return  somme


def entropies(w,q):
	
	entrops=[]
	for i,wi in w.items():
	
			entrops.append((i, entropie(wi,q)))
			
	
	return entrops
	


w=poid(Occ,21)
list_entropie = entropies(w,21)

entropie_best = sorted(list_entropie, key=lambda t:t[1], reverse=True)[:3]


def argmaxi(wi):
	
	maxi=0
	maxA='/'
	
	for a,wia in wi.items():
	
		if(maxi<wia):
			maxi=wia
			maxA=a

	return a


acid_dmove=[]

for e in entropie_best:
	
	acid_dmove.append(argmaxi(w[e[0]]))
	
print acid_dmove




"""
plt.plot([x[0] for x in list_entropie], [x[1] for x in list_entropie])
plt.show()
"""	

def fzero(b, w):
	somme = 0
	for i, wi in w.items():
		somme += wi[b]
		
	return float(somme)/L
"""
def pzero(w):
	pdt = 1
	for a in A:
		pdt *= fzero(a, w)
	
	return pdt
"""
#equation 9
def test_sequence_l(w, B, k):
	#tester si une sequence de L proteine de B appartient a la famille donnee
	#calcul la proba quela sequence fasse partie de la famille
	#B sequence a tester
	somme = 0
	for i, wi in w.items():
		
		bi = B[i+k]
		
		somme+=math.log(float(wi[bi])/fzero(bi,w),2)
		
	return somme
	
	
def vraisem_bi(w, B):
	liste_proba = []
	for k in range(0, N-L):
		liste_proba.append((k, test_sequence_l(w, B, k)))
	return liste_proba



proba_seq = vraisem_bi(w, B)
plt.plot([x[0] for x in proba_seq], [x[1] for x in proba_seq])
plt.show()


		

