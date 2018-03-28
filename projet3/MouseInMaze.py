from CdM import CdM


class MouseInMaze(CdM):


	def __init__(self):

		self.states=[1,2,3,4,5,6]
		self.avancer=0.25
		self.reculer=0.25
		self.rester=0.5

	def get_states(self):
  
  		return self.states
  		
  	
  	def get_transition_distribution(self, state):
  	
  		if(state==1):
  			return {"1":0.5, "2":0.5}
  			
  		if(state==2):
  			return {"1":0.5, "4":0.5}
  			
  		if(state==3):
  			return {"1":0.25, "5":0.25, "2":0.25, "6":0.25}
  			
  		if(state==4):
  			return {"3":1}
  			
  		if(state==5):
  			return {"5":1}
  			
  		if(state==6):
  			return {"6":1}
  		
  	
  	def get_initial_distribution(self):
  		
  		self.state=2
  		self.t=0
