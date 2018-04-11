# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

import matplotlib.pyplot as plt
from matplotlib import colors

import utils


class CdM(object):
	"""
	Class virtuelle représentant une Chaîne de Markov
	"""

	def __init__(self):
		"""
		Constructeur. En particulier, initalise le dictionaire stateToIndex

		:warning: doit être appelé en fin de __init__ des classes filles
		avec ` super().__init__()`
		"""
		self.state=self.get_states()
		self.stateToIndex=dict()

		for i in range(len(self.state)):
			self.stateToIndex[self.state[i]]=i


	def get_states(self):
		"""
		:return: un ensemble d'états énumérable (list, n-uple, etc.)
		"""
		raise NotImplementedError

	def get_transition_distribution(self, state):
		"""
		:param state: état initial
		:return: un dictionnaire {etat:proba} représentant l'ensemble des états atteignables à partir de state et leurs
		probabilités
		"""
		raise NotImplementedError

	def get_initial_distribution(self):
		"""
		:return: un dictionnaire représentant la distribution à t=0 {etat:proba}
		"""
		raise NotImplementedError


	def __len__(self):
		"""
		permet d'utiliser len(CdM) pour avoir le nombre d'état d'un CdM

		:warning: peut être surchargée
		:return: le nombre d'état
		"""
		return len(self.get_states())

	def show_transition_matrix(self):
		utils.show_matrix(self.get_transition_matrix())
	
	
	def distribution_to_vector(self, distrib):
	
		vect=np.zeros(len(self.state))
		
		for (state, proba) in distrib.items():

				vect[self.stateToIndex[state]]=proba
		
		return vect
		
		
	def vector_to_distribution(self, vector):
		
		distrib=dict()
		
		for (state, index) in self.stateToIndex.items():
			
			if(vector[index]!=0.0):
				distrib[state]=vector[index]
				
				
		return distrib
		
		
	
	def show_distribution(self, init_distrib):
		vect = self.distribution_to_vector(init_distrib)

		# c = np.zeros((100, 4))
		# c[:, -1] = np.linspace(0, 1, 100)  # gradient de transparence
		# for i in range(len(vect)):
		# 	c[:,i] = vect[i]
		# # ProbaMap est une matplotlib.colormap qu'on utilisera pour afficher des valeurs de probabilités (de 0 à 1).
		# ProbaMap = colors.ListedColormap(c)
		
		# plt.matshow(vect, cmap=ProbaMap)
		# plt.grid(False)
		# plt.show()
		size = len(self.get_states())
		fig, ax = plt.subplots()
		fig.set_size_inches(4, 1)
		ax.set_yticks([])
		ax.set_xticklabels(self.get_states())
		ax.set_xticks([i for i in range(size)])
		ax.imshow(self.distribution_to_vector(init_distrib).reshape(1, size), cmap=utils.ProbaMap)


	def get_transition_matrix(self):

		matrix =[]

		for (state, index) in self.stateToIndex.items():

			matrix.append( self.distribution_to_vector(self.get_transition_distribution(state)))

		return np.array(matrix)

	def get_transition_graph(self):
		#creer un graph oriente
		g = gum.DiGraph()
		#creer autant de noeuds qu'il y a d'etats
		for i in range(len(self.state)):
			# g.addNodeWithId(i+1)
			g.addNode()

		#recup la matrice de transitions
		matrix = self.get_transition_matrix()
		#ajouter les transitions au graph
		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				c = matrix[i][j]
				if(c > 0):
					g.addArc(i, j)
		
		return g
	
	def show_transition_graph(self, g):
		g.showDot(self.get_transition_graph().toDot())
		# print (g)
    
	def get_communication_classes(self):
		graph = self.get_transition_graph()
		nodes = graph.ids()

		# limite le nombre de tests en passant les elements deja parcourus par profondeur
		used = set()

		classes = []
		for i in nodes:
			if not i in used:
				children = graph.children(i)
				visites = set()
				connexes = set()
				visites.add(i)
				connexes.add(i)
				self.profondeur(graph, children, visites, connexes, classes, used)

		return classes

	# parcours d'une branche en profondeur
	def profondeur(self, graph, children, visites, connexes, classes, used):
		for j in children:
			j_children = graph.children(j)
			if j in visites:
				if j in connexes:
					# si j a ete visite et marque comme connexe:
					#	ajouter ses parents aussi si ce n'est pas encore fait
					for p in graph.parents(j).intersection(visites):
						connexes.add(p)
						used.add(p)
				else:
					# s'il a un de ses fils dans connexes alors il devrait le rejoindre
					if len(set(j_children).intersection(connexes))>0:
						connexes.add(j)
						used.add(j)
						for p in graph.parents(j):
							connexes.add(p)
							used.add(p)
			else:
				# si pas encore visite marquer comme tel et parcourir ses enfants en profondeur
				visites.add(j)
				self.profondeur(graph, j_children, visites, connexes, classes, used)
	
		#ajouter la nouvelle classe connexes si enrichissante
		if len(classes) > 0:
			c = classes.pop()#recup le dernier element ajoute
			if len(c.intersection(connexes))>0:
				# enrichir les classes existantes si possible
				connexes = c.union(connexes)
			else:
				#sinon remettre c dedans
				classes.append(c)
		#dans tous les cas ajouter connexes
		classes.append(connexes)
	
	# recup des sous-chaines de Markov
	def get_absorbing_classes(self):
		graph = self.get_transition_graph()
		classes_connexes = self.get_communication_classes()
		absorbants = []
		for c in classes_connexes:
			children = set()
			for i in c:
				children = children.union(graph.children(i))
			diff = children.difference(c)
			if len(diff) == 0:
				absorbants.append(c)
		return absorbants
	
	def is_irreducible(self):
		states = set(self.stateToIndex.values())
		graphe = self.get_transition_graph()
		print (states)
		for i in states:
		
			visites=self.profondeur_irreducible(graphe, i, set([i]))
			print(i, " ", visites)
			if len(states.difference(visites))>0:
				
				return False
				
		return True	
		
	#fonction auxiliaire de is_irreducible
	def profondeur_irreducible(self, graph, i, visites):
		children = graph.children(i)
		for c in children:
			if c in visites:
				continue
			visites.add(c)
			visites.union(self.profondeur_irreducible(graph, c, visites))
			
		return visites
	
	def is_aperiodic(self):
		if self.is_irreducible():
			return True
		
	
