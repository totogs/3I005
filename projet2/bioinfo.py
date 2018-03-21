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
			#print data[m,i]
			dict_occ[i][data[m,i]]+=1

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

	return maxA


acid_dmove=[]

for e in entropie_best:
	
	#print argmaxi(w[e[0]])
	acid_dmove.append(argmaxi(w[e[0]]))
	
print acid_dmove
print entropie_best



"""

plt.plot([x[0] for x in list_entropie], [x[1] for x in list_entropie])
plt.title("entropie relative en fonction de la position")
plt.xlabel("position")
plt.ylabel("entropie")
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


"""
proba_seq = vraisem_bi(w, B)

print "w0('-') = ", w[0]["-"]
print "S0 = ", list_entropie[0]
print "l(b0,...,bL-1) =", proba_seq[0]

plt.plot([x[0] for x in proba_seq], [x[1] for x in proba_seq])
plt.title("Vraisemblance en fonction de la position")
plt.xlabel("position")
plt.ylabel("log-vraisemblance")
plt.show()

"""

#partie 2


#nij (a, b)

	

def nij(i,j,a,b,data):

	cpt=0
	
	for k in range(0,M):
	
		if(data[k,i]==a and data[k,j==b]):
			cpt+=1
			
	return cpt
	
	
def wij(i,j,a,b,q,data):

	return (nij(i,j,a,b,data) + 1.0/q)/(M+q)
	


def Mij(i,j,q,data):
	
	som=0
	
	for a in A:
		for b in A:
		
			wijAB=wij(i,j,a,b,q,data)	
			
			som+=wijAB*math.log(wijAB/(w[i][a]*w[j][b]),2)
			
	return som
			
			
			
def matrice_mut(data,q):
	
	mat=[]
	for i in range(L):
		temp=[]
		for j in range(L):
			
			print(i,j)
			if(i==j):
				temp.append(0)
			else:
				temp.append(Mij(i,j,q,data))
			
		mat.append(temp)
		
	return mat
	
	
Mat_mut=matrice_mut(Dtrain,21)

print Mat_mut
		
		
			
		
		

