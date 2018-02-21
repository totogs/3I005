import email
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def read_file(fname):
    """ Lit un fichier compose d'une liste de emails, chacun separe par au moins 2 lignes vides."""
    f = open(fname,'rb')
    raw_file = f.read()
    f.close()
    raw_file = raw_file.replace(b'\r\n',b'\n')
    emails =raw_file.split(b"\n\n\nFrom")
    emails = [emails[0]]+ [b"From"+x for x in emails[1:] ]
    return emails

def get_body(em):
    """ Recupere le corps principal de l'email """
    body = em.get_payload()
    if type(body) == list:
        body = body[0].get_payload()
    try:
        res = str(body)
    except Exception:
        res=""
    return res

def clean_body(s):
    """ Enleve toutes les balises html et tous les caracteres qui ne sont pas des lettres """
    patbal = re.compile('<.*?>',flags = re.S)
    patspace = re.compile('\W+',flags = re.S)
    return re.sub(patspace,' ',re.sub(patbal,'',s))
    
def get_emails_from_file(f):
	mails = read_file(f)
	return [ s for s in [clean_body(get_body(email.message_from_bytes(x))) for x in mails] if s !=""]
        
spam = get_emails_from_file("spam.txt" )
nospam = get_emails_from_file("nospam.txt")

"""
for s in spam:
	print (s)
"""

def split(liste, x):
	l1 = []
	l2 = []
	pivot = math.floor(len(liste)*x)
	
	l1 = liste[0:pivot]
	l2 = liste[pivot:]
	
	return l1, l2



#np.bin

def longueur_body(em):

	return len(em)
	
#print(longueur_body(l1[0]))

#pour calculer l'histogramme des longeur de mails
def liste_longueur(lem):
	li=[]
	for l in lem:
		li.append(longueur_body(l))
		
	return li
"""
liste = liste_longueur(l1)
length = len(liste)
plt.hist(liste,bins=int(length/20))
"""
#Q 2.3
def apprend_modele(spam, non_spam):
	#renvoie la proba qu'un email soit d'une longueur donnee sachant que c'est un spam
	#p(X=x | Y=+1) = p(Y=+1 | X=x) * p(X=x) / p(Y=+1)
	#or p(Y=+1) = 0.5
	#et p(X=x) = nbr email de longueur x / nbr email
	#et p(Y=+1 | X=x) = nbr email spam de taille x / nbr longueur de taille x
	
	#renvoyer la distribution des spam selon leur longueur xfrom numpy import linalg as LA
	
	#calculs:
	#suppression des doublons dans les listes
	liste_mails = liste_longueur(list(set(spam+non_spam)))
	#tableau (longueur, proba)
	dict_lp = []#dictionnaire longueur, proba spam
	
	for x in liste_mails:
		dict_lp.append((x,distribution(spam, non_spam, x)))
	
	return dict_lp
	


def distribution(spam, non_spam, x):
	#renvoie p(X=x | Y=+1) pour une longueur x donnee
	nb_x_spam = 0 #nbre de spam de longueur x
	nb_x_tot = 0 #nbre total de mail de longueur x
	
	for lm in liste_longueur(spam):
		if (lm == x):
			nb_x_spam += 1
			nb_x_tot += 1
	for lm in liste_longueur(non_spam):
		if (lm == x):
			nb_x_tot += 1
	
	px = float(nb_x_tot) / (nb_x_tot + nb_x_spam) #p(X=x)
	pyx = float(nb_x_spam) / nb_x_tot #p(Y=+1 | X=x)
	
	pxy = pyx * px / 0.5 #p(X=x | Y=+1)
	
	return pxy
	
def predict_email(emails, modele):
	#renvoie la liste des labels pour l'ensemble des emails en fonction du modele passe en parametre

	labels = [] #labels[i] contient le label de l'email emails[i]
	#emails contient la longueur des emails
	for e in emails:
		
		proba=0.5
		for m in modele:
			if(longueur_body(e)>=m[0] and longueur_body(e)<m[1]):
				proba=m[2]
		
		if (proba > 0.5):
			labels.append(+1)
		else:
			labels.append(-1)
	
	return labels
	

#Q 2.4 P(f(x) = y)
def accuracy(emails, modele):
	#emails[i] = (email, label)
	list_email=[]
	
	for e in emails:
		list_email.append(e[0])

	labels=predict_email(list_email,modele)
	cpt=0.0
	
	for i in range(len(labels)):
		if(labels[i]*emails[i][1]>=0):
			cpt+=1.0
	
	return cpt/len(labels)
	
	
	
def proba_err(emails,modele):

	return (1.0-accuracy(emails,modele))



#Renvoit la liste des probabilités d'un spam en fonction d'intervalles de longueur de mail
def regroup(modele, bins):
	#la longueur du plus long mail
	
	modele=sorted(modele,key=lambda model: model[0], reverse=True)
	
	new_modele=[]
	proba=1.0
	l_max = modele[0][0]
	step=int(l_max/bins)
	
	cpt=0;
	
	for i in range(0, l_max,step):
			
			for m in modele:
				
				if(m[0]>=i and m[0]<i+step):
					proba*=m[1]
					cpt+=1
					
			if(cpt>0):
				new_modele.append((i,i+step,proba))
			
			else:
				new_modele.append((i,i+step,0.5))
				
			proba=1.0
			cpt=0
	
	return new_modele
		
		

l1_s,l2_s=split(spam, 0.5)
l1_ns,l2_ns=split(nospam, 0.5)	
	
	
modele=apprend_modele(l1_s,l1_ns)

emails=[]

for l in l2_s:
	emails.append((l,+1))
for l in l2_ns:
	emails.append((l,-1))

"""	
modele_bine=regroup(modele,len(modele)/1000)

	
print(proba_err(emails,modele_bine))
"""


def apprend_modeleSem(spam,nospam):

	mails = ' '.join(list(set(spam+nospam)))
	liste_mots=list(set(mails.split()))

	reg=r"[0-9_@\\\/]+"
	
	#tableau (longueur, proba)
	dictionnaire= []#dictionnaire mot, proba spam
	print(len(liste_mots))
	for mot in liste_mots:
		if(re.match(reg,mot)is None and len(mot)<27 and len(mot)>3):
			
			dictionnaire.append((mot,distributionSem(spam, nospam, mot)))
	
	return dictionnaire

	

def distributionSem(spam,nospam, xi):
    
	
	nb_xi_spam=0#Nombre de fois ou le mot xi apparait dans les spams
	nb_xi_tot=0#Nombre de fois ou le mot xi apparait dans les mails


	for mail in spam:
		for mot in mail.split():

			if (mot.lower() == xi.lower()):
				nb_xi_spam += 1
				nb_xi_tot += 1

	for mail in nospam:
		for mot in mail.split():

			if (mot.lower() == xi.lower()):
				nb_xi_tot += 1
	

	px = float(nb_xi_tot) / (nb_xi_tot + nb_xi_spam) #p(X=xi)
	pyx = float(nb_xi_spam) / nb_xi_tot #p(Y=+1 | X=xi)

	pxy = pyx * px / 0.5 #p(X=xi | Y=+1)

	return pxy
	

def predict_emailSem(email,modele):

	
	X=[]
	proba=0.5

	for mot in modele:
		
		if(email.find(mot[0])!=-1):
			proba*=mot[1]


	if(proba>0.5):
		return +1
	else:
		return -1

liste_x=[]

def accuracySem(emails, modele):

	labels=[]
	
	for e in emails:
		labels.append(predict_emailSem(e[0],modele))
	cpt=0.0
	
	for i in range(len(labels)):
		if(labels[i]*emails[i][1]>=0):
			cpt+=1.0
	
	return cpt/len(labels)
	


def proba_errSem(emails, modele):

    return (1.0-accuracySem(emails,modele))


modele_sem=apprend_modeleSem(l1_s, l1_ns)

print(proba_errSem(emails,modele_sem))



def distance(xi,xj):

	return -np.dot(xi,xj)/(LA.norm(xi)*LA.norm(xj))

def Pij(listeX, i, j):
	
	somme=0.0
	for k in range(len(listeX)):
		if(k!=i):
			somme+=math.exp(-distance(listeX[i],listeX[k])/(2*np.var(listeX[i])))
			
	math.exp(-distance(listeX[i],listeX[j])/(2*np.var(listeX[i])))/somme
	

def Qij(listeX, i, j):
	
	somme=0.0
	for k in range(len(listeX)):
		if(k!=i):
			somme+=math.exp(-distance(listeX[i],listeX[k]))
			
	math.exp(-distance(listeX[i],listeX[j]))/somme
	

