###### 	import modules BEGIN ######
import os 
###### limit the number of threads #######
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

import numpy as np				# Array 
import pdb 					# Debugging 
import sys					# Get parameter from command line
import scipy.io as sio				# Read Matlab files
from pathlib import Path			# Path for files
import time					# Record simulation time

from CommandLineParamatize import ComdPara 	# Self-defined Function to paramatize the passed variables from command line
from DataInput import DataInput			# self-defined Function to read the input data

from gp_hawkes_PostGapFi_fxTau_changePriDeltaWalk import gp_hawkes_PostGapFi_fxTau_changePriDeltaWalk
##### import modules END   ######

###### parameterize the command line BEGIN ######
Dic_ComdPara = ComdPara(sys.argv);
globals().update( Dic_ComdPara );

np.random.seed(123);
List_Seed = np.random.permutation(10000)
np.random.seed(List_Seed[int(SimTime)-1])


###### #################################   ######

###### read Unobserved & observed data  ######

Dic_InputData = DataInput(UnbDataPath);
#globals().update( DataInput(UnbDataPath, ObDataPath) );

###### ########################### ######

###### Define the Object to Call the Methods  ######

class MAB_Method_Dispach:

	def __init__ (self, Method):		# get name of the method
		# "Initial MAB_Method_Dispach: Call Discpatch() to get the corresponding MAB Simulation Object"
		self.method = getattr(self, Method, lambda: "Invalid Method Input");
	
	def Dispatch(self, Dict_MAB_Var):			# Discpach the method
		return self.method(Dict_MAB_Var);

	###### Define Methods  ######
	def HpSpUCB(self, Dict_MAB_Var):
		MAB = HpSpUCB( Dict_MAB_Var );
		return MAB;
	
	###### ############### ######

####################################################


checkPath = ResultsPath.replace("/scratch/chiangwe/2020_02_28_HawkesProcessSimulation/results/",\
				"/home/chiangwe/PhD2018/2020_02_28_HawkesProcessSimulation/lock_")

isExist = os.path.exists(checkPath) 
#isExist = False;
sim_times = int(SimTime)+1

if not isExist:
	Dic_ComdPara.update( {'RandSeed':str(List_Seed[sim_times-1])} )
	Dic_ComdPara.update( {'SimTime':str(sim_times)} )
	Dic_ComdPara.update( {'ResultsPath':ResultsPath} )
	Dict_MAB_Var = {};
	Dict_MAB_Var.update( Dic_InputData );
	Dict_MAB_Var.update( Dic_ComdPara );

	#################################################

	MAB = MAB_Method_Dispach(Method).Dispatch( Dict_MAB_Var );
	MAB.simulation()
	
	#################################################
	FileId = open(ResultsPath, "r");
	content = FileId.readlines()
	content = content[-1];
	FileId.close()
	#################################################
	FileId = open(checkPath, "w+");
	FileId.write(content)
	FileId.close()
exit()
###### ##################### ######

############ Back Up
'''
from epi_greedy import epi_greedy				# Method Epi-geedy algorithm 
#from epi_greedy_explr_ob import epi_greedy_explr_ob 		# Method Epi-greedy Explr Ob events algorithm

#from gaussian_process import gaussian_process			# Method Gaussian process with online algorithm
from gaussian_process_george_sim import gaussian_process_george_sim
from gaussian_process_online import gaussian_process_online 	# Method Gaussian process algorithm
from gaussian_process_online_sim import gaussian_process_online_sim

from epi_greedy_hawkes import epi_greedy_hawkes # Method Epi-greed algorithm but exploit highest intensity
from epi_greedy_hawkesSumExp import epi_greedy_hawkesSumExp # Method Epi-greed algorithm but exploit highest intensity Using Sum of Exponential model
from epi_greedy_hawkesSumExpSelect import epi_greedy_hawkesSumExpSelect  # Method Epi-greed algorithm but exploit highest intensity Using Sum of Exponential model Give many choices and use likelidhood to choose
from epi_greedy_hawkesSumExpEplrSim import epi_greedy_hawkesSumExpEplrSim 
#
from lin_ucb import lin_ucb			# Method LinUCB contextual MAB
from lin_ucbsim import lin_ucbsim
from ucb1 import ucb1
from ucb1_hawkes import ucb1_hawkes
from ucb1_hawkes_SameWeight import ucb1_hawkes_SameWeight
from ucb1_hawkes_sim import ucb1_hawkes_sim
from ucb1_hawkesCI import ucb1_hawkesCI
from ucb1_hawkes_MultiSim import ucb1_hawkes_MultiSim

from ucb1_hawkes_CIsimNum import ucb1_hawkes_CIsimNum
from ucb1_hawkes_SPsimNum import ucb1_hawkes_SPsimNum

from ucb1_hawkes_intensity import ucb1_hawkes_intensity
#
'''

'''

	def EpiGreedy(self, Dict_MAB_Var):
		MAB = epi_greedy( Dict_MAB_Var );
		return MAB;

	### Variation of Epi greedy algorithm ###

	### In stead of random exploration, explore the ARMs with max avg Obs events 
	def EpiGreedyExplrOb(self, Dict_MAB_Var): 

		MAB = epi_greedy_explr_ob( Dict_MAB_Var );
		return MAB;

	#########################################
	
	def GaussianProcess(self, Dict_MAB_Var):
		MAB = gaussian_process( Dict_MAB_Var );
		return MAB;
	def GaussianProcessGeorgeSim(self, Dict_MAB_Var):
		MAB = gaussian_process_george_sim( Dict_MAB_Var );
		return MAB;
	def GaussianProcessOnline(self, Dict_MAB_Var):
		MAB = gaussian_process_online( Dict_MAB_Var );
		return MAB;
	def GaussianProcessOnlineSim(self, Dict_MAB_Var):
		MAB = gaussian_process_online_sim( Dict_MAB_Var );
		return MAB;
	def EpiGreedyHawkes(self, Dict_MAB_Var):
		MAB = epi_greedy_hawkes( Dict_MAB_Var );
		return MAB;
	def EpiGreedyHawkesSumExp(self, Dict_MAB_Var):
		MAB = epi_greedy_hawkesSumExp( Dict_MAB_Var );
		return MAB;
	def EpiGreedyHawkesSumExpSelect(self, Dict_MAB_Var):
		MAB = epi_greedy_hawkesSumExpSelect( Dict_MAB_Var );
		return MAB;
	def EpiGreedyHawkesSumExpEplrSim(self, Dict_MAB_Var):
		MAB = epi_greedy_hawkesSumExpEplrSim( Dict_MAB_Var );
		return MAB;
	def UCB1(self, Dict_MAB_Var):
		MAB = ucb1( Dict_MAB_Var );
		return MAB;
	def UCB1Hawkes(self, Dict_MAB_Var):
		MAB = ucb1_hawkes( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesIntensity(self, Dict_MAB_Var):
		MAB = ucb1_hawkes_intensity( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesSameWeight(self, Dict_MAB_Var):
		MAB = ucb1_hawkes_SameWeight( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesSim(self, Dict_MAB_Var):
		MAB = ucb1_hawkes_sim( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesCI(self, Dict_MAB_Var):
		MAB = ucb1_hawkesCI( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesCIgSim(self, Dict_MAB_Var):
		MAB = ucb1_hawkesCIgSim( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesCIMultiSim(self, Dict_MAB_Var):
		MAB = ucb1_hawkes_MultiSim( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesCIsimNum(self, Dict_MAB_Var):
		MAB = ucb1_hawkes_CIsimNum( Dict_MAB_Var );
		return MAB;
	def UCB1HawkesSPsimNum(self, Dict_MAB_Var):
		MAB = ucb1_hawkes_SPsimNum( Dict_MAB_Var );
		return MAB;
	def LinUCB(self, Dict_MAB_Var):
		MAB = lin_ucb( Dict_MAB_Var );
		return MAB;
	def LinUCBSim(self, Dict_MAB_Var):
		MAB = lin_ucbsim( Dict_MAB_Var );
		return MAB;
'''
