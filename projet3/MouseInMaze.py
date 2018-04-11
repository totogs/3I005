from CdM import CdM


class MouseInMaze(CdM):


	def __init__(self):
		self.avancer=0.25
		self.reculer=0.25
		self.rester=0.5
		super().__init__()

	def get_states(self):
		return [1,2,3,4,5,6]
	

	def get_transition_distribution(self,state):
		if(state==1):
			return{1: 0.5, 2: 0.5}

		if(state==2):
			return{1: 0.5, 4: 0.5}

		if(state==3):
			return{1: 0.25, 5: 0.25, 2: 0.25, 6: 0.25}

		if(state==4):
			return{3: 1}

		if(state==5):
			return{5: 1}

		if(state==6):
			return{6: 1}


	def get_initial_distribution(self):

		#self.state=2
		#self.t=0
		#ons uppose qu'on commence tjrs a l'etat 2
		return {2: 1}
	
