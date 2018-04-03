from MouseInMaze import MouseInMaze
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

from IPython.display import display, Image

m = MouseInMaze ()
# m.show_transition_matrix()
m.show_transition_graph(gnb)
m.get_communication_classes()