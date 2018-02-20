# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:11:51 2018

@author: 3505137
"""

import random
import numpy as np
import matplotlib.pyplot as plt



def paquet():
	cartes = []	
	for i in range(1,14):
		for c in ['C', 'K', 'P', 'T']:
			cartes.append((i, c))
	random.shuffle(cartes)
	return cartes
	
def meme_position(p, q):
	l = []	
	for i in range(len(p)):
		if p[i] == q[i]:
			l.append(i)
			
	return l

p = paquet()
q = paquet()

#print meme_position(p,q)

#la probabilité d'obtenir la meme carte à l'indice i est 1/52
def exp(i, n):
	cpt = 0
	for j in range(n):
		p1 = paquet()
		p2 = paquet()
		if p1[i] == p2[i]:
			cpt+=1
			
	return float(cpt)/n

	
#print exp(4, 1000)


def moyenne_carte_pos(n):
	cpt=0.0
	for i in range(n):
		p1=paquet()
		p2=paquet()
		cpt+=len(meme_position(p1,p2))
		
	return cpt/n
	
#print moyenne_carte_pos(1000)


def evolution(n,pas):
	
	l=[]
	j=[]
	
	for i in range(1,n,pas):
		j.append(i)
		l.append(moyenne_carte_pos(i))
	
	
	plt.plot(j,l)
	
	
#evolution(10000,100)
	
#la moyenne tourne autour de 1
	
	
my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list, 1):
    print(c, value)
	
	