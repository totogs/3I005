# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
import functools

c = np.zeros((100, 4))
c[:, -1] = np.linspace(0, 1, 100)  # gradient de transparence
c[:, 0] = 0.3  # rouge
c[:, 1] = 0.5  # vert
c[:, 2] = 0.5  # bleu
# ProbaMap est une matplotlib.colormap qu'on utilisera pour afficher des valeurs de probabilités (de 0 à 1).
ProbaMap = colors.ListedColormap(c)


def show_matrix(matrix):
  """
  :warning: ne devrait pas être surchargé
  présente la matrice de transition
  :param matrix: np.array qui devrait être une matrice stochasique
  """
  plt.matshow(matrix, cmap=ProbaMap)
  plt.grid(False)
  plt.show()


def pgcd(a, b):
  """pgcd(a,b): calcul du 'Plus Grand Commun Diviseur' entre les 2 nombres entiers a et b"""
  while b != 0:
    a, b = b, a % b

  return a

def pgcd_list(L):
  if len(L) == 0:
      return 1
  return functools.reduce(pgcd, L)

# parcours d'une branche en profondeur
def profondeur(graph, children, visites, connexes, classes, used):
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
      profondeur(graph, j_children, visites, connexes, classes, used)

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

#fonction auxiliaire de is_irreducible
def profondeur_irreducible(graph, i, visites):
  children = graph.children(i)
  for c in children:
    if c in visites:
      continue
    visites.add(c)
    visites.union(profondeur_irreducible(graph, c, visites))
    
  return visites

def period_elem(graph, root, children, visited, periods, d):
		# print("children ", children)
		if root in children:
			# on est revenu a la racine
			periods.add(d)
			return periods
		for child in children:
			if child in visited:
				return periods
			visited.add(child)
			periods.union(period_elem(graph, root, graph.children(child), visited, periods, d+1))
		
		return periods