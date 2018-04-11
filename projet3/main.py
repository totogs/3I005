from MouseInMaze import MouseInMaze
from MonoBestiole import MonoBestiole
from FeuRouge import FeuRouge
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

from IPython.display import display, Image

m = MouseInMaze ()
# m.show_transition_matrix()
# m.show_transition_graph(gnb)
print("connexes: ", m.get_communication_classes())
print("absorabants: ",m.get_absorbing_classes())
print("Irréductible : "+str(m.is_irreducible()))
# print (list(set([[1, 2], [1, 2], [2, 1]])))
feu = FeuRouge()
print("connexes: ",feu.get_communication_classes())
print("absorbants: ",feu.get_absorbing_classes())
print("Irréductible : "+str(feu.is_irreducible()))
# feu.show_transition_matrix()



