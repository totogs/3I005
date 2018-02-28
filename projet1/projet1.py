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
	dict_lp = []#dictionnaire longueur, proba spam, et proba d'une longueur x'

	for x in liste_mails:
		pxy_s, pxy_ns = distribution(spam, non_spam, x)

		dict_lp.append((x, pxy_s, pxy_ns))

	return dict_lp



def distribution(spam, non_spam, x):
	#renvoie p(X=x | Y=+1) pour une longueur x donnee
	nb_x_spam = 0.001 #nbre de spam de longueur x
	nb_x_nospam = 0.001 #nbre de non spam de longueur x
	nb_x_tot = 0.001 #nbre total de mail de longueur x

	for lm in liste_longueur(spam):
		if (lm == x):
			nb_x_spam += 1
			nb_x_tot += 1
	for lm in liste_longueur(non_spam):
		if (lm == x):
			nb_x_nospam +=1
			nb_x_tot += 1

	px = float(nb_x_tot) / (len(spam)+len(non_spam)) #p(X=x)
	pyx_s = float(nb_x_spam) / nb_x_tot #p(Y=+1 | X=x)
	pyx_ns = float(nb_x_nospam) / nb_x_tot #p(Y=-1 | X=x)
	

	pxy_s = pyx_s * px / 0.5 #p(X=x | Y=+1)
	pxy_ns = pyx_ns * px / 0.5 #p(X=x | Y=-1)
	

				
	return -math.log(pxy_s), -math.log(pxy_ns)

def predict_email(emails, modele):
	#renvoie la liste des labels pour l'ensemble des emails en fonction du modele passe en parametre

	labels = [] #labels[i] contient le label de l'email emails[i]
	#emails contient la longueur des emails
	for e in emails:
		#pr chaque longueur si p(X=x | Y=+1) > p(X=x | Y=-1) -> c'est un spam
		proba=0.5
		for m in modele:
			#m : [bonrne_inf, borne_sup, p(X=x | Y=+1), p(X=x | Y=-1)]
			#on est dans la bonne tranche?
			if(longueur_body(e)>=m[0] and longueur_body(e)<m[1]):
				pxy_s=m[2]
				pxy_ns=m[3]


				if (pxy_s > pxy_ns):
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



#Renvoit la liste des probabilites d'un spam en fonction d'intervalles de longueur de mail
def regroup(modele, bins):
	#la longueur du plus long mail
	#tri par ordre decroissant des longueurs
	modele=sorted(modele,key=lambda model: model[0], reverse=True)
	

	new_modele=[]
	proba_pxy_s=1.0
	proba_pxy_ns=1.0
	l_max = modele[0][0]
	step=int(l_max/bins)

	cpt=0;

	for i in range(0, l_max,step):
			#pour chaque tranche de longueurs entre i et i+step
			#on calcul p_spam, p_nospam sur l'inrevalle
			for m in modele:
				
				if(m[0]>=i and m[0]<i+step):
					#si la probaba est nulle on la prend pas en compte_mot_email
					#m[1] : p(X=x | Y=+1)
					#m[2] : p(X=x | Y=-1)
					if m[1] == 0 or m[2] == 0:
						continue
					else:
						proba_pxy_ns+=m[2]
						proba_pxy_s+=m[1]
						cpt+=1
			#si au moins un elements dans l'intervalle
			if(cpt>0):
				new_modele.append((i,i+step,proba_pxy_s,proba_pxy_ns))
			#si aucun element dans cet intervalle -> une chance sur deux
			else:
				new_modele.append((i,i+step,0.5, 0.5))

			proba=1.0
			cpt=0
	

	#renvoyer le modele regroupe
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
modele_bine=regroup(modele,10000)

print("Pourcentage de données mal classées ")
print(proba_err(emails,modele_bine))

"""

#Exercice 3: Classification a partir du contenu d'un email 

"""
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
"""

def apprend_modeleSem(spam,nospam):
	

	liste_mots = filtrer(' '.join(list(set(spam+nospam))))
	liste_spam = (' '.join(spam)).split()
	liste_nospam = (' '.join(nospam)).split()
	"""
	liste_mots = filtrer_nltk(' '.join(list(set(spam+nospam))))

	"""
	liste_mots = compte_mot_email(liste_mots, set(spam+nospam))


	#tableau (longueur, proba)
	dictionnaire= []#dictionnaire mot, proba spam
	print(len(liste_mots))
	print(len(liste_spam))
	print(len(liste_nospam))


	for mot in liste_mots:
		p = distributionSem(liste_spam, liste_nospam, mot)
		if(p!=-1):
			dictionnaire.append((mot,p))
	
	print(dictionnaire)
	return dictionnaire

	

def filtrer(mails):

	
	liste_mots=list(set(mails.split()))
	reg=r"[0-9_@\\\/]+"

	new_list=[]

	for mot in liste_mots:

		if(re.match(reg,mot) is None and len(mot)<27 and len(mot)>3):
			new_list.append(mot)


	return new_list


"""
def filtrer_nltk(mails):

	stop_words = set(stopwords.words('english'))
	stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
	words = list(set(word_tokenize(mails)))
	words = nltk.pos_tag(words)

	reg=r"[0-9_@\\\/]+" 

	new_words = []

	tags = ["SYM","LS","CD", "CC", "PRP", "PRP$", "TO","FW" ,"EX", ]

	for word in words:
		
		if ((word[0].lower() not in stop_words) and (word[1] not in tags) and len(word[0])>3 and len(word[0])<27 and re.match(reg,word[0]) is None):
			new_words.append(WordNetLemmatizer().lemmatize(word[0],'v'))
	 

	return new_words;
"""

def compte_mot_email(liste_mots, mails):

	freq = []

	for mot in liste_mots:
		cpt=0
		for mail in mails:

			if(mail.find(mot)!=-1):
				cpt+=1
		
		if(cpt > 3):
			freq.append(mot)

	return freq


def distributionSem(spam,nospam, xi):
    
	
	nb_xi_spam=0#Nombre de fois ou le mot xi apparait dans les spams
	nb_xi_tot=0#Nombre de fois ou le mot xi apparait dans les mails


	for mot in spam:

		if (mot.lower() == xi.lower()):
			nb_xi_spam += 1
			nb_xi_tot += 1

	for mot in nospam:

		if (mot.lower() == xi.lower()):
			nb_xi_tot += 1
	
	if(nb_xi_tot==0):
		
		return -1

	px = float(nb_xi_tot) / (nb_xi_tot + nb_xi_spam) #p(X=xi)
	pyx = float(nb_xi_spam) / nb_xi_tot #p(Y=+1 | X=xi)

	pxy = pyx * px  #p(X=xi | Y=+1)

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



#Renvoie le pourcentage de mails bien classes par rapport a des donnees d'entrainement
def accuracySem(emails, modele):

	labels=[]
	
	for e in emails:
		labels.append(predict_emailSem(e[0],modele))
	cpt=0.0
	
	for i in range(len(labels)):
		if(labels[i]*emails[i][1]>=0):
			cpt+=1
	
	return float(cpt)/len(labels)
	

#Renvoie la probabilite de l'erreur c'est a dire 1 moins la probabilite de de mails bien classes
def proba_errSem(emails, modele):

    return (1.0-accuracySem(emails,modele))


"""
modele_sem = apprend_modeleSem(l1_s, l1_ns)
print(proba_errSem(emails,modele_sem))
"""



#Partie 2: Visualisation
def distance(xi,xj):

	return -np.dot(xi,xj)/(LA.norm(xi)*LA.norm(xj))

#modelisation de la probabilite que la representation Xi soit dans le voisinage de Xj
# plus d(Xi, Xj) est grande et plus Pij est faible
def Pij(listeX, i, j):

	somme=0.0
	for k in range(len(listeX)):
		if(k!=i):
			somme+=math.exp(-distance(listeX[i],listeX[k])/(2*np.var(listeX[i])))

	math.exp(-distance(listeX[i],listeX[j])/(2*np.var(listeX[i])))/somme

#modelisation de la probabilite que la representation Yi soit dans le voisinage
#dans l'espace de faible dimension
def Qij(listeX, i, j):

	somme=0.0
	for k in range(len(listeX)):
		if(k!=i):
			somme+=math.exp(-distance(listeX[i],listeX[k]))

	math.exp(-distance(listeX[i],listeX[j]))/somme

#on veut que Qij soit la plus proche de Pij pour conserver le meme voisinage pour
#les deux representations Xi et Yi
# --> minimiser une distance entre les deux distributions de probabilite
#divergence de Kullback-Leiber, calculant la distance entre Pi et Qi
def KL(listeX, i):
    #listeX : un vecteur de longueurs de mails
    # somme sur j des Pij * log(Pij/Qij), i!=j
    somme = 0
    for j in range(len(listeX)):
        if i == j:
            continue
        else:
            pij = Pij(listeX, i, j)
            qij = Qij(listeX, i, j)
            somme += pij * math.log(pij/qij)

    return somme

#retrouvons pour chaque email i sa representation Yi qui minise la somme des
#distances entre Pi et Qi

#on effecue une descente de gradient en utilisant les derivees partielles de la fonction a minimiser

def C(listeX):
    # C = sum i (sum j!=i Pij * log (Pij / Qij))
    #   = sum i KL(listeX, i)
    summe = 0
    for i in range(len(listeX)):
        somme += KL(listeX, i)
#derivees partielles
def dCy1(listeX, listeY, i):
    #listeX : longueur des email i
    #listeY : vecteurs Yi = (yi1, yi2), representations des emails i
    #les Yi sont initialises aleatoirement pour chaque email i selon la loi N(0, 0.5)

    somme = 0
    for j in range(len(listeY)): #len(listeX) = len(listeY)
        pij = Pij(listeX, i, j)
        pji = Pij(listeX, j, i)
        qij = Qij(listeX, i, j)
        qji = Qij(listeX, j, i)
        yi1 = listeY[i][0]
        yj1 = listeY[j][0]

        somme += (yi1 - yj1)*(pij - qij + pji - qji)
    return 2*somme

def dCy2(listeX, listeY, i):
    #listeX : vecteurs X(x0, x1, .... xn) -> xi nombre d'apparition du mot a l'indice i du dico
    #listeY : vecteurs Yi = (yi1, yi2), representations des emails i
    #les Yi sont initialises aleatoirement pour chaque email i selon la loi N(0, 0.5)

    somme = 0
    for j in range(len(listeY)): #len(listeX) = len(listeY)
        pij = Pij(listeX, i, j)
        pji = Pij(listeX, j, i)
        qij = Qij(listeX, i, j)
        qji = Qij(listeX, j, i)
        yi1 = listeY[i][1]
        yj1 = listeY[j][1]

        somme += (yi1 - yj1)*(pij - qij + pji - qji)
    return 2*somme

def SNE(listeX):
	#initilsation aleatoire de listeY
	listeY = [np.random.normal(0, 0.5, size=2) for x in listeX]
	#epsilon choisie a la main entre 10e-1 et 10e-3
	epsilon = 0.01
	#repeter jusqu'a convergence
	#dans notre cas on repete l'operation n fois avec n >> 2 * len(listeX) pour etre sur d'avoir un bon resultat
	cpt = 0
	while(cpt < 4 * len(listeX)):
		cpt+=1
		#tirer un email au hasard
		i = np.random.randint(0, high = len(listeX))
		#mettre a jour Yi
		y1 = listeY[i][0]
		y2 = listeY[i][1]

		listeY[i][0] = y1 - epsilon*dCy1(listeX, listeY, i)
		listeY[i][1] = y2 - epsilon*dCy2(listeX, listeY, i)

	return listeY
	
	

def create_vectX(dictionnaire, email):
	#renvoie un vecteur X avec le nombre d'apparition de chaque mot du dico
	liste = []
	
	for mot in dictionnaire:
		
		cpt=0
		
		for i in email[0].split():
		
			if(mot[0].lower()== i.lower()):
				
				cpt+=1
		
		liste.append(cpt)
	
	return liste


def create_listeX(dictionnaire, emails):
	
	listex = []
	
	for e in emails:
	
		listex.append(create_vectX(dictionnaire, e))
		
	return listex
	

print(emails[0])	
dictionnaire = filtrer(' '.join(list(set(spam+nospam))))	
dictionnaire = compte_mot_email(dictionnaire, set(spam+nospam))

listeX = create_listeX(dictionnaire, emails)
print(listeX)
