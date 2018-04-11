# -*- coding: utf-8 -*-
from Collector import Collector
import numpy as np


class CollGetDistribution(Collector):
	def __init__(self, epsilon, pas):
		self.epsilon=epsilon
		self.pas=pas
		self.dico=dict()
		self.iterations=1
		self.erreur=0.0

	def initialize(self, cdm, max_iter):

		pass
	def receive(self, cdm, iter, state):
		
		if(iter>0):	
			self.distribformer=np.divide(cdm.distribution_to_vector(self.dico),iter)
		else:
			self.distribformer=cdm.distribution_to_vector(self.dico)
		
		if(state not in self.dico):
			self.dico[state]=1.0
		else:
			self.dico[state]+=1
			
		
		self.iterations=iter+1
		self.distribnew=np.divide(cdm.distribution_to_vector(self.dico),self.iterations)
		
		self.erreur=np.amax(np.absolute(self.distribformer-self.distribnew))
		
		if(self.erreur<self.epsilon):
			return True
		
		
		if(iter%self.pas==0):
			cdm.show_distribution(self.dico)
		
		return False

	def finalize(self, cdm, iteration):
	
		for state, nbr in self.dico.items():

			self.dico[state]=nbr/self.iterations
			
		

	def get_results(self, cdm):
		return {"erreur": self.erreur, "proba":self.dico}
