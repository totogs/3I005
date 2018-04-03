from CdM import CdM

class MonoBestiole(CdM):
    def __init__(self, nb_states, p, q):
        #p: proba d'aller a droite
        #q: proba d'aller a gauche
        self.states=[i+1 for i in range(nb_states)]
        self.droite = p
        self.gauche = q
        super().__init__()
    
    def get_states(self):
        return self.states
    
    def get_transition_distribution(self,state):
        if(state == 1):
            return {1: 0.4, 2: 0.6}
        elif(state == len(self.states)):
            return {state: 0.6, state-1: 0.4}
        else:
            return {state-1: 0.4, state+1: 0.6}

    def get_initial_distribution(self):
        return {1: 1}
    