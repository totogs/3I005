from MouseInMaze import MouseInMaze
from MonoBestiole import MonoBestiole
from FeuRouge import FeuRouge
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

from IPython.display import display, Image

m = MouseInMaze ()
# m.show_transition_matrix()
# m.show_transition_graph(gnb)
m.get_communication_classes()

# print (list(set([[1, 2], [1, 2], [2, 1]])))
feu = FeuRouge()
feu.get_communication_classes()
# feu.show_transition_matrix()