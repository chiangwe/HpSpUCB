###### import modules BEGIN ######
import numpy as np                              # Array
import sys
from sys import getsizeof
import os
import glob
from os import path
import subprocess
import psutil
process = psutil.Process(os.getpid())
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from scipy.sparse import csc_matrix		# Sparse Array
import pdb                                      # Debugging
import math					# Mathematical functions
from collections import Counter			# Count Frequency
from scipy.sparse import csr_matrix, lil_matrix	# Make Sparse Matrix
from scipy.ndimage.filters import gaussian_filter
import time
#from statsmodels.stats.weightstats import DescrStatsW
import gc
from multiprocessing import Process, Queue
from sklearn.metrics.pairwise import manhattan_distances
#
# Tick Hawkes process package
from scipy.interpolate import Rbf
import hashlib
#from filelock import Timeout, FileLock, SoftFileLock
from tick.hawkes import SimuHawkes, HawkesExpKern, HawkesKernelTimeFunc, HawkesKernelExp, HawkesEM, HawkesSumExpKern, SimuHawkesSumExpKernels, SimuHawkesExpKernels
from tick.base import TimeFunction
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
# from hawkeslib import UnivariateExpHawkesProcess as UVHP

###### import modules END   ######

class HpSpUCB:

	def __init__(self, Dict_MAB_Var) :
		
		####### Construct the variables #######
		
		#### Check the passed variables ####

		Dict_MAB_Var_Keys = [*Dict_MAB_Var.keys()]; # keys of MAB_Var

		List_var = [\
			'UnbData', 		# Unobeservation data, ground Truth (311, OD ...)
			'NumPulledArm',		# Number of arms pulled in each time
			'PeriodPerPull',	# Time interval for each action (pull)
			'NumArm',		# Number of arms
			'Method',		# Name of the Method
			'PathNum',
			# Method Specific 
			'Gamma',	# Combine two scores
			'Lambda',	# Kernel
			'Tau',		# prob
			'AlphaR',	#
			'AlphaH',
			'Delta',
			#
			'nsample',
			'ResultsPath',           # Save results
			'SimTime',
			'RandSeed'
			 ];

		# check whether the required var exist
		Check_exist = [(item, item in Dict_MAB_Var_Keys) for item in List_var];
		No_required = list( filter(lambda tup: False in tup, Check_exist) );

		if len(No_required) > 0:
			print("\n\nNeed the fllowing Variables\n Exit ...", No_required);
			exit();
			# Cast the variables in to types

		### ########################### #### 

		#### Cast the passed variables ####
		self.UnbData = Dict_MAB_Var.pop('UnbData');
		
		self.NumPulledArm = int( Dict_MAB_Var.pop('NumPulledArm') ); 	# integer number
		self.PeriodPerPull = float( Dict_MAB_Var.pop('PeriodPerPull') );# float number 
		self.NumArm = int( Dict_MAB_Var.pop('NumArm') ); # int number 
		self.Method = Dict_MAB_Var.pop('Method');
		self.SimTime = int(Dict_MAB_Var.pop('SimTime'))
		self.Tmax = self.UnbData[-1,2]
		self.RandSeed = int( Dict_MAB_Var.pop('RandSeed') );
		self.Grand  = int(math.sqrt(self.NumArm))
		self.PathNum = Dict_MAB_Var.pop('PathNum')
		
		# Open resutls file for write
		self.ResultsPath = Dict_MAB_Var.pop('ResultsPath');
		self.Results_WriteOut = open(self.ResultsPath, 'w+', buffering=1)
		self.Results_WriteOut.write("Iteration\tScore\tArmID\tAvgNumEvents\tExplore\n")

		# Method Specific
		self.Gamma = float(Dict_MAB_Var.pop('Gamma'))
		self.Lambda =  float(Dict_MAB_Var.pop('Lambda'))
		self.Tau = float(Dict_MAB_Var.pop('Tau'))
		self.AlphaR = float(Dict_MAB_Var.pop('AlphaR'))
		self.AlphaH = float(Dict_MAB_Var.pop('AlphaH'))
		self.Delta = float(Dict_MAB_Var.pop('Delta'))
		# 
		self.nsample = int(Dict_MAB_Var.pop('nsample'))

		#### Define the variables for simulation ####

		self.TimeStmp = 0; 							# Initial simulation timestamp
		self.TimeStmpLast = self.TimeStmp;
		self.NumPulls = math.ceil( self.Tmax/self.PeriodPerPull );    	 	# Number of actions (pulls) in total simulation
		
		self.PulledArm = np.random.choice( self.NumArm, self.NumPulledArm, replace=False )	# Initialization of Pulled Arms
		self.CurrentIdx_Unb = 0;
		self.NextIdx_Unb = 0;
		
		self.CurrentIdx_Ob = 0;
		self.NextIdx_Ob = 0;

		#### Define the variable for Epigreedy ####
		
		self.RecrdTimesArms = np.zeros((self.NumArm, ));
		self.RecrdAvgScrArms = np.zeros((self.NumArm, ));
		self.RecrdAvgNumArms = np.zeros((self.NumArm, ));
		self.RecrdAvgIntenseArms = np.zeros((self.NumArm, ));
		
		self.SimUpfront = int( ((self.NumArm / self.NumPulledArm)) )
		self.ListofStepRcd = [ [None] for _ in range(self.NumArm)]
		self.ListofStepEvNumRcd = [ [0] for _ in range(self.NumArm)]
		
		self.UpfronRecord = [ None for _ in range(self.NumArm)]
		self.UpfronRecord_InputSave = [None for _ in range(self.NumArm)];
		self.ListofParameter = [ [None] for _ in range(self.NumArm)] # It may need to be stored
		#
		self.PullProb  = np.array(np.zeros(self.NumPulledArm,));
		self.Hawkes_Score_out = np.array(np.zeros(self.NumPulledArm,));
		self.Hawkes_Score = np.array(np.zeros(self.NumArm,));
		self.ScoreCb = np.array(np.zeros(self.NumPulledArm,));
		#
		self.Explor = 'N'
		### ################################## ####

		#### Define the variable for Gaussian Process  ####
		
		self.pred = np.zeros((self.NumArm, ));       # prediction mean
		self.sigma = np.zeros((self.NumArm, ));      # prediction variance
		self.queryProb = np.ones((self.NumArm, ))*1/self.NumArm; # Probibiliaty for Arm
		self.TrainDataX = np.zeros((0, 2));
		self.TrainDataY = np.zeros((0, ));
		
		# Feature for All arms
		self.RequesArray = np.unravel_index( np.array( range(0, self.NumArm)),  (self.Grand,self.Grand) );
		self.RequesArray = np.hstack((np.array( self.RequesArray[0],ndmin=2 ).transpose(), np.array( self.RequesArray[1],ndmin=2 ).transpose()));
		self.RequesArray = np.matrix( self.RequesArray )

		##########################################
		self.Sim_Tuple = [];
		self.PulledCheck = np.zeros( ( self.NumArm, 1) );
		self.PulledCheck[ self.PulledArm ] = 1;
		### ############################### ####

		# Score Record
		self.Score = np.zeros( (self.NumPulls, ) ); 		# Recort Score after each pull
		self.Accu = 0;						# Accumulative score for each pull
		self.AccuScore = np.zeros((self.NumPulls,  ) );		# Recort Accumulative score for each pull
		#### ################################### ####
		
		######################################################################################
		self.TimeStampPath = '/scratch/chiangwe/2019_10_10_HawkesProcessSimulation/results/' + \
						self.Method + '/temp/' + str(os.getpid()) + '_ListofTimeStamp_Arm_';
		self.TimeStampUpfrontPath = '/scratch/chiangwe/2019_10_10_HawkesProcessSimulation/results/' + \
						self.Method + '/temp/' + str(os.getpid()) + '_TimeStampUpfront_Arm_';
		self.IntenseUpfrontPath = '/scratch/chiangwe/2019_10_10_HawkesProcessSimulation/results/' + \
						self.Method + '/temp/' + str(os.getpid()) + '_IntenseUpfront_Arm_';
		######################################################################################
		
		for ar in range(0, self.NumArm):
			TimeStampPath = self.TimeStampPath+ str(ar) + ".pkl";
			TimeStampUpfrontPath = self.TimeStampUpfrontPath + str(ar) + ".pkl";
			IntenseUpfrontPath = self.IntenseUpfrontPath + str(ar) + ".pkl";
			
			f = open( TimeStampPath, 'wb'); pickle.dump( [ None ], f); f.close();
			f = open( TimeStampUpfrontPath, 'wb'); pickle.dump( [ None ], f); f.close();
			f = open( IntenseUpfrontPath, 'wb'); pickle.dump( [ None ], f); f.close();
		
		######################################################################################
	# Hawkes MLE
	def HawkesExp(self, t_in, StartTime, EndTime):
		ts_in = t_in  - StartTime;
		EndTime_temp =  EndTime - StartTime;
		#
		decays_list = [ [[10.0**ep]] for ep in np.arange(-8, 2,1)]
		baseline_list = [];
		adjacency_list = [];
		LikeScore_list = [];
		for decays in decays_list:
			learner = HawkesExpKern(decays, penalty='l2', C=1, gofit='least-squares')
			learner.fit( [ts_in] )
			baseline_list.append( learner.baseline )
			adjacency_list.append( learner.adjacency )
			LikeScore_list.append( learner.score() )
		#pdb.set_trace()
		#
		IdSelect = np.argsort( np.array(LikeScore_list) )[::-1][0];
		baseline = baseline_list[IdSelect][0].tolist()
		adjacency = adjacency_list[IdSelect][0][0].tolist()
		decays = decays_list[IdSelect][0][0].tolist()
		LikeScore = LikeScore_list[IdSelect]
		#
		Intensity = baseline + np.array( [  decays * adjacency * math.exp( -decays*(EndTime_temp-t)) for t in ts_in ] ).sum()
		return baseline, adjacency, decays, Intensity.tolist(), LikeScore
	
	# Hawkes Simulation and Get the best Timestamp based on Likehood
	def HawkesLHPick(self, Ts_Candidate, Ts_NewObs, BaselineStartTime, SimStartTime, SimEndTime, paraTuples):
		
		Ts_NewObs = np.array( Ts_NewObs );

		# Calculate the Baseline Function From SimStartTime to SimEndTime
		LikelihoodDiff = []
		# Simulated Time Series 
		for Ts_Observed, paras in zip( Ts_Candidate, paraTuples):
			
			# Get parameter
			Baseline = paras[0];
			Alpha = paras[1];
			Decay = paras[2];
			
			Ts_with_Observed = np.array( Ts_Observed );
			Ts_upto_NewObs = np.hstack( (Ts_with_Observed, Ts_NewObs) )
			# 
			# re position 
			Ts_with_Observed = Ts_with_Observed - BaselineStartTime
			Ts_upto_NewObs = Ts_upto_NewObs - BaselineStartTime
			
			# Likelihood Before
			EndTimeBefore = SimEndTime - BaselineStartTime;
			learner = HawkesExpKern(decays=Decay, penalty='l1', C=20, gofit='likelihood')
			try:
				fit_score_Before = learner.score(events=[Ts_with_Observed], end_times=EndTimeBefore, baseline=np.array([Baseline]),\
							 adjacency=np.array([[Alpha]]) )
			except:
				pdb.set_trace()
				print( Ts_with_Observed, EndTimeBefore)
				fit_score_Before = learner.score(events=[Ts_with_Observed], end_times=EndTimeBefore, baseline=np.array([Baseline]),\
                                                         adjacency=np.array([[Alpha]]) )
			# Likelihood After
			EndTimeAfter= SimEndTime + self.PeriodPerPull - BaselineStartTime;
			learner = HawkesExpKern(decays=Decay, penalty='l1', C=20, gofit='likelihood')
			try:
				fit_score_After = learner.score(events=[Ts_upto_NewObs], end_times=EndTimeAfter, baseline=np.array([Baseline]),\
							adjacency=np.array([[Alpha]]) )
			except:
				pdb.set_trace()
				print( Ts_with_Observed, EndTimeBefore)
				fit_score_After = learner.score(events=[Ts_upto_NewObs], end_times=EndTimeAfter, baseline=np.array([Baseline]),\
                                                        adjacency=np.array([[Alpha]]) )
			# Likelihood From Simulated Data
			deltaLikeHood = fit_score_After - fit_score_Before;
			LikelihoodDiff.append( deltaLikeHood )
		Idx = np.argmax(np.array( LikelihoodDiff) )	
		TimestampBest = Ts_Candidate[Idx]
		# Where BestPara is not useful
		return TimestampBest, Idx
	
	# Save
	def Savetemp(self, Path_Temp, Input):
		f = open( Path_Temp, 'wb')
		pickle.dump( Input, f, protocol=2)
		f.close()
	
	def Loadtemp(self, Path_Temp):
		f = open( Path_Temp, 'rb')
		Output = pickle.load(f); 
		f.close();
		if len(Output) == 2:
			return Output[0], Output[1]
		else:
			return Output
	
	# define soft max to transform mean value to probability
	def softmax(self, x):
		orig_shape = x.shape
		# Vector
		x = x/self.Tau;
		x_max = np.max(x);
		x = x - x_max;
		numerator = np.exp(x)
		denominator =  1.0 / np.sum(numerator)
		x = numerator.dot(denominator)
		return x
		
	def simulation(self):
		
		####### Simulation Loop #######
		# Record Time
		start = time.time()
		start_total = time.time()
		
		Hawkes_mean = np.zeros((1,100))[0].tolist();
		Hawkes_std = np.zeros((1,100))[0].tolist();
		
		for step in range(0, self.NumPulls):
			
			# Move the time stamp
			self.TimeStmp = self.TimeStmp + self.PeriodPerPull;

			# Calculate the Index for next 
			DataPtsMove = np.searchsorted(self.UnbData[ self.CurrentIdx_Unb:,2], self.TimeStmp, 'right');
			self.NextIdx_Unb = self.CurrentIdx_Unb + DataPtsMove;

			# Counting Dictionary for Unb, Ob, and Dicover
			self.CtDictUnb = dict( Counter( self.UnbData[ self.CurrentIdx_Unb:self.NextIdx_Unb ,3] ) );
			self.CtDicDisc = {a:self.CtDictUnb[a] for a in self.PulledArm if a in [*self.CtDictUnb.keys()]}

			# Epi-Greedy algorithm : calculate the mean and the times that Arms have been pulled
			PulledArmCnt = np.array( [ self.CtDicDisc.setdefault(item,0) for item in self.PulledArm]  );
			PulledArmCnt_update = np.multiply( self.RecrdAvgScrArms[self.PulledArm], self.RecrdTimesArms[self.PulledArm] ) + \
						PulledArmCnt;
			
			self.RecrdTimesArms[self.PulledArm] = self.RecrdTimesArms[self.PulledArm] + 1;
			self.RecrdAvgScrArms[ self.PulledArm ] = np.divide( PulledArmCnt_update, self.RecrdTimesArms[ self.PulledArm ] );

			# Record the Score
			self.Accu = self.Accu + np.sum( PulledArmCnt );
			self.Score[step] = np.sum( PulledArmCnt );
			self.AccuScore[step] = self.Accu;
			#
			# Memory Usage:
			MEMUSAGE = process.memory_info().rss / 34359738368;
			if MEMUSAGE > 0.50:
				self.Results_WriteOut.write("MEMOUT: "+ MEMUSAGE + "\n");
				exit()
			end = time.time()
			
			##########################################################################
			# get hawkes score
			#########################################################################
			Hawkes_mean_out = [Hawkes_mean[arm] for arm in self.PulledArm ];
			Hawkes_std_out = [Hawkes_std[arm] for arm in self.PulledArm ];
			Hawkes_mean_out = " ".join( ["{:.6e}".format(st) for st in Hawkes_mean_out ] );
			Hawkes_std_out =  " ".join( ["{:.6e}".format(st) for st in Hawkes_std_out ]  );
			##########################################################################
			
			
			# Record the results
			String_out = "{:d}".format( int(self.AccuScore[step]) )  + "\t" + \
			             ",".join( [ "{:3d}".format( arm ) for arm in self.PulledArm ] ) + "\t" + str(step)+\
				      "\t" + "{:.4f}".format(MEMUSAGE*100) + "%" +"\t"+\
					"{:.4f}".format( (start-start_total)/3600) +"\t"+Hawkes_mean_out+"\t"+Hawkes_std_out+"\n";
			
			# Get Start Time
			start = time.time()
			
			self.Results_WriteOut.write(String_out);
			print( String_out )
			
			if(step == self.NumPulls-1):
				break;
			
			######################################################################################
			#self.UCB1 = np.sqrt( 2 * math.log( step+2 ) / self.RecrdTimesArms );
			# Gather dataset for gaussian process
			TrainData = np.unravel_index( np.array([*self.PulledArm]),  (self.Grand, self.Grand) );
			TrainData = np.vstack((TrainData[0], TrainData[1]) ).T
			self.TrainDataX = np.vstack( (self.TrainDataX, TrainData) );
			self.TrainDataY = np.hstack( (self.TrainDataY, PulledArmCnt) );
			
			# Define Kernel
			kernelRBF = kernels.RBF( self.Lambda )
			gp1 = GaussianProcessRegressor(kernel=kernelRBF, random_state=self.RandSeed )
			gp1.fit( self.TrainDataX, self.TrainDataY)
			self.pred, self.sigma = gp1.predict( self.RequesArray, return_std=True)
			
			self.query = self.pred + self.AlphaR * self.sigma;
			#
			if self.query[~np.isinf( self.query )].sum() != 0:
				self.query_norm = self.query / self.query[~np.isinf(self.query)].sum()
			else:
				self.query_norm = self.query
			#self.queryProb = self.softmax(self.query);

			# Fix when num of nonzero prob is less than numPullArm
			#if( sum( self.queryProb != 0 ) < self.NumPulledArm):
			#	IdxNonZero = np.where( self.queryProb!=0 );
			#	IdxZero = np.where( self.queryProb==0 );
			#	NumZero = IdxZero[0].shape[0];
			#	self.queryProb[ IdxZero ] = 10**(-20);
			#	IdxMax = np.argmax(self.queryProb);
			#	self.queryProb[IdxMax] = self.queryProb[IdxMax] - 10**(-20) * NumZero;
			
			#############################################################################################
			# Hawkes Process
			# Get Time Stamps for Discovered Events and Count the number of Discovered Events
			ArmTime = self.UnbData[ self.CurrentIdx_Unb:self.NextIdx_Unb ,2:4]
			ArmTime = ArmTime[:,::-1]
			DicrTimeSeries = [ ArmTime[ np.where(ArmTime[:,0] == Arm),1].ravel() for Arm in self.PulledArm ]
			DicrTimeSeries_num = [arr.shape[0] for arr in DicrTimeSeries]
			Evoke_Hawkes = [ num>0 for num in DicrTimeSeries_num ];
			
			for Ev, ts, ar in zip(Evoke_Hawkes, DicrTimeSeries, self.PulledArm):

				TimeStampPath = self.TimeStampPath+ str(ar) + ".pkl";
				TimeStampUpfrontPath = self.TimeStampUpfrontPath + str(ar) + ".pkl";
				IntenseUpfrontPath = self.IntenseUpfrontPath + str(ar) + ".pkl";
				
				# Record Each Step for Arms
				self.ListofStepRcd[ar].append(step)
				
				# Record Number of Events for pullsed Arms at step
				self.ListofStepEvNumRcd[ar].append( ts.shape[0] )
				
				#############################################################
				# Oberseve the first one with timestamps in it.
				############################################################
				# Note: We take the Estimated Intensity Sampled from Posterior Dist.
				if ( Ev>0 ) & ( np.array( self.ListofStepEvNumRcd[ar][:-1] ).sum() == 0 ):
					
					###############################################################
					# This is the first time to observe cells with timestamps
					# Estime MLE parameter with Timestamp, starttime, endtime
					# Estimation time cover the timespan in timeseries
					# StartTime -- ts -- EndTime
					###############################################################
					
					StartTime = self.ListofStepRcd[ar][1] * self.PeriodPerPull;
					EndTime = (step + 1)* self.PeriodPerPull;
					
					# Estimate Parameters for Prior Distribution
					Pri_Mu, Pri_Alpha, Pri_Beta, Pri_Intese, Pri_LikeHood = self.HawkesExp( ts, StartTime, EndTime );
					
					# Save Time Stamp for Sampling from Posterior 
					self.Savetemp( TimeStampPath, ts.tolist() );
					
					# Draw Samples From Posterior
					command = "python2 -W ignore ./UVHP_ChangePriDeltaWalk.py Path_Temp=" + TimeStampPath + " Seed_in=" + str(step) +\
					" StartTime=" + str(StartTime) + " EndTime="+str(EndTime) +" Delta="+ "{:.8f}".format(self.Delta) +\
					" Pri_Mu=" + "{:.8f}".format(Pri_Mu) + " Pri_Alpha=" + "{:.8f}".format(Pri_Alpha) +\
					" Pri_Beta=" + "{:.8f}".format(Pri_Beta) + " n_sample=" + str(self.nsample);
					p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
					(output, err) = p.communicate()
					p_status = p.wait()
					
					# Kill Process
					PidCheck = int( output.decode().split('\n')[1].split("=")[1] )
					if psutil.pid_exists(PidCheck):
						psutil.Process( PidCheck ).terminate()
					else:
						pass;
					
					results = [ [ float(temp) for temp in st.split("\t")] for st in output.decode().split('\n')[2:-1] ]
					
					print("MLE: ", Pri_Mu, Pri_Alpha, Pri_Beta )
					for temp in set(output.decode().split('\n')[2:-1]):
						print(temp.split("\t"))
					# Parse Sampled Parameters
					self.ListofParameter[ar] = [ (results[Idx][0], results[Idx][1], results[Idx][2]) \
						for Idx in range( 0, len(results) ) ]
					
					# Baseline Rate Can't be zero
					if Pri_Mu == 0:
						Pri_Mu = 10**-8;
					
					######################################################################
					# Simulated Time Series Upfront
					# We simulated SimUpfront time period
					# BaselineStartTime -- ts -- SimStartTime -- SimUpFront -- SimEndTime
					# Update Parameters
					# Use Estimated
					######################################################################
					
					
					# Simulate some timestamps upfront
					BaselineStartTime = self.ListofStepRcd[ar][1] * self.PeriodPerPull
					SimStartTime = (self.ListofStepRcd[ar][-1] + 1) * self.PeriodPerPull
					SimEndTime = SimStartTime +  self.PeriodPerPull * self.SimUpfront
					
					# if The Simulation EndTime is larger than total Observed Time 
					if (self.NumPulls+1) * self.PeriodPerPull < SimEndTime:
						simEndTime = (self.NumPulls+1) * self.PeriodPerPull;
					
					# Save Simulation End Time
					self.UpfronRecord[ar] = SimEndTime;
					
					# Save Parameters that we use to simulate upfront timestamp
					self.UpfronRecord_InputSave[ar] = \
						( BaselineStartTime, SimStartTime, SimEndTime, self.ListofParameter[ar], step )
					
					# Save timestamp and list of parameters from posterior 
					self.Savetemp( TimeStampPath, [ ts, self.ListofParameter[ar] ] );
					
					command = "./python3 -W ignore ./HawkesSImV2.py" + \
					" Path_Temp=" + TimeStampPath + " Seed=" + str(step) + " BaselineStartTime=" + str(BaselineStartTime) +\
					" SimStartTime=" + str(SimStartTime) + " SimEndTime=" + str(SimEndTime)+ \
					" PeriodPerPull=" + str(self.PeriodPerPull) + " n_simTime=" + str(1)+\
					" Path_TimeStampTemp=" + TimeStampUpfrontPath + " Path_IntenseUpTemp=" + IntenseUpfrontPath;
					p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
					(output, err) = p.communicate()
					p_status = p.wait()
					
					# Kill Process
					PidCheck = int( output.decode().split('\n')[1].split("=")[1] )
					if psutil.pid_exists(PidCheck):
						psutil.Process( PidCheck ).terminate()
					else:
						pass;
					
					ISMEMOUT = (output.decode('utf8').split("\n")[-2].split(" ")[0] == 'MEMOUT')
					if ISMEMOUT:
						print(  output.decode('utf8').split("\n")[-2] )
						self.Results_WriteOut.write( output.decode('utf8').split("\n")[-2] + "\n");
						exit()
				
				#############################################################
				# Oberseve and There is some timetemps in it already.
				# 	Get Parameters and Simulate the timestamps. 
				#	Choose the one has largest LH 
				############################################################
				
				elif ( np.array( self.ListofStepEvNumRcd[ar][:-1] ).sum() != 0 ):
					
					##################################################################################################
					# Simulate the TimeStamp
					# BaselineStartTime -- Ts_Observed -- SimStartTime -- NotPulled -- SimEndTime -- ts -- ObserveTime
					# Check Upfont Covers or Not
					# BaselineStartTime -- 
					##################################################################################################
					
					BaselineStartTime = self.ListofStepRcd[ar][1] * self.PeriodPerPull
					SimStartTime = (self.ListofStepRcd[ar][-2] + 1 ) * self.PeriodPerPull
					SimEndTime = (self.ListofStepRcd[ar][-1] ) * self.PeriodPerPull
					ObserveTime = SimEndTime + self.PeriodPerPull;
					Ts_NewObs = ts
					
					# Load All simulated upfront timestamps and Retain before SimEndTime
					# If simulated timestamp upfront covers, pick time stamp before SimEndTime
					if self.UpfronRecord[ar] >= SimEndTime:
						
						f = open( TimeStampUpfrontPath, 'rb'); 
						ListSimulateTimeStamp = pickle.load(f); f.close();
						
						ListSimulateTimeStamp =\
						[ t_temp[ np.where(t_temp < SimEndTime)] for t_temp in ListSimulateTimeStamp ]
						
					else:
					# If Not
						
						command = "./python3 -W ignore ./HawkesSImV2.py" + \
						" Path_Temp=" + TimeStampPath + " Seed=" + str(step) + " BaselineStartTime=" + str(BaselineStartTime) +\
						" SimStartTime=" + str(SimStartTime) + " SimEndTime=" + str(SimEndTime)+ \
						" PeriodPerPull=" + str(self.PeriodPerPull) + " n_simTime=" + str(1)+\
						" Path_TimeStampTemp=" + TimeStampUpfrontPath + " Path_IntenseUpTemp=" + IntenseUpfrontPath;
						p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
						(output, err) = p.communicate()
						p_status = p.wait()
						
						
						# Kill Process
						PidCheck = int( output.decode().split('\n')[1].split("=")[1] )
						if psutil.pid_exists(PidCheck):
							psutil.Process( PidCheck ).terminate()
						else:
							pass;
						
						ISMEMOUT = (output.decode('utf8').split("\n")[-2].split(" ")[0] == 'MEMOUT')
						if ISMEMOUT:
							print(  output.decode('utf8').split("\n")[-2] )
							self.Results_WriteOut.write( output.decode('utf8').split("\n")[-2] + "\n");
							exit()
				
						#
						self.UpfronRecord[ar] = SimEndTime
						#
						# Update the parameter
						tuplePara = self.UpfronRecord_InputSave[ar];
						self.UpfronRecord_InputSave[ar] = \
							 ( tuplePara[0], tuplePara[1], SimEndTime, tuplePara[3], tuplePara[4] );

						f = open( TimeStampUpfrontPath, 'rb');
						ListSimulateTimeStamp = pickle.load(f); f.close();
					
					# Pick the best Timestamp Based on loglikelihood
					tuplePara = self.UpfronRecord_InputSave[ar];
					
					TimestampBest, BestIdx = \
					self.HawkesLHPick( ListSimulateTimeStamp, Ts_NewObs, \
							BaselineStartTime, SimStartTime, SimEndTime, tuplePara[3])
					# Update TimeSeries
					# Save Time Stamp for Sampling from Posterior 
					self.Savetemp( TimeStampPath, np.hstack( (TimestampBest,ts) ).tolist() );
					
					####################################################################################################
					# ReEstimate
					# # BaselineStartTime -- Ts_Observed -- SimStartTime -- NotPulled -- SimEndTime -- ts -- ObserveTime
					# Estime MLE parameter with Timestamp, starttime, endtime
					####################################################################################################
					
					Pri_Mu, Pri_Alpha, Pri_Beta, Pri_Intese, Pri_LikeHood = \
						self.HawkesExp( np.hstack( (TimestampBest,ts) ), BaselineStartTime, ObserveTime);
					
					if Pri_Mu == 0:
						Pri_Mu = 10**-8;
					
					command = "python2 -W ignore ./UVHP_ChangePriDeltaWalk.py Path_Temp=" + TimeStampPath + " Seed_in=" + str(step) +\
					" StartTime=" + str(BaselineStartTime) + " EndTime="+str(ObserveTime) +" Delta="+ "{:.8f}".format(self.Delta) +\
					" Pri_Mu=" + "{:.8f}".format(Pri_Mu) + " Pri_Alpha=" + "{:.8f}".format(Pri_Alpha) +\
					" Pri_Beta=" + "{:.8f}".format(Pri_Beta) + " n_sample=" + str(self.nsample);
					p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
					(output, err) = p.communicate()
					p_status = p.wait()
					
					# Kill Process
					PidCheck = int( output.decode().split('\n')[1].split("=")[1] )
					if psutil.pid_exists(PidCheck):
						psutil.Process( PidCheck ).terminate()
					else:
						pass;
					results = [ [ float(temp) for temp in st.split("\t")] for st in output.decode().split('\n')[2:-1] ]

					
					print( "MLE: ", Pri_Mu, Pri_Alpha, Pri_Beta )
					for temp in set(output.decode().split('\n')[2:-1]):
						print(temp.split("\t"))
					####################################################################
					# Simulated Time Series Upfront
					# Use Estimated
					# Simulate some timestamps upfront
					# ObserveTime -- SimUpFront
					###################################################################
					
					self.ListofParameter[ar] = [ (results[Idx][0], results[Idx][1], results[Idx][2]) \
						for Idx in range( 0, len(results) ) ]
					
					SimStartTime = ObserveTime
					SimEndTime = SimStartTime +  self.PeriodPerPull * self.SimUpfront
					if (self.NumPulls+1) * self.PeriodPerPull < SimEndTime:
						simEndTime = (self.NumPulls+1) * self.PeriodPerPull;
					
					self.UpfronRecord[ar] = SimEndTime;
					#
					self.UpfronRecord_InputSave[ar] = \
                                                ( BaselineStartTime, SimStartTime, SimEndTime, self.ListofParameter[ar], step )
					# Save timestamp and list of parameters from posterior
					self.Savetemp( TimeStampPath, [ np.hstack( (TimestampBest,ts) ).tolist(), self.ListofParameter[ar] ] );
					
					command = "./python3 -W ignore ./HawkesSImV2.py" + \
					" Path_Temp=" + TimeStampPath + " Seed=" + str(step) + " BaselineStartTime=" + str(BaselineStartTime) +\
					" SimStartTime=" + str(SimStartTime) + " SimEndTime=" + str(SimEndTime)+ \
					" PeriodPerPull=" + str(self.PeriodPerPull) + " n_simTime=" + str(1)+\
					" Path_TimeStampTemp=" + TimeStampUpfrontPath + " Path_IntenseUpTemp=" + IntenseUpfrontPath;
					p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
					(output, err) = p.communicate()
					p_status = p.wait()
					
					# Kill Process
					PidCheck = int( output.decode().split('\n')[1].split("=")[1] )
					if psutil.pid_exists(PidCheck):
						psutil.Process( PidCheck ).terminate()
					else:
						pass;
					ISMEMOUT = (output.decode('utf8').split("\n")[-2].split(" ")[0] == 'MEMOUT')
					if ISMEMOUT:
						print(  output.decode('utf8').split("\n")[-2] )
						self.Results_WriteOut.write( output.decode('utf8').split("\n")[-2] + "\n");
						exit()
						
			##########################################################
			# Get Intensity From Upfront 
			##########################################################
			UpIntense = [];
			currentstep = step;
			ObserveTime = (step + 1)* self.PeriodPerPull
			
			for ar in range( 0, self.NumArm):
				
				TimeStampPath = self.TimeStampPath+ str(ar) + ".pkl";
				TimeStampUpfrontPath = self.TimeStampUpfrontPath + str(ar) + ".pkl";
				IntenseUpfrontPath = self.IntenseUpfrontPath + str(ar) + ".pkl";
				
				# If There is an Simulated Upfront SImeEndTime
				if  (self.UpfronRecord[ ar ]!= None):
					# If Simulated Upfront DID NOT covers ObserveTime 
					if ( self.UpfronRecord[ ar ] <= ObserveTime ):
						
						# Simulated Upfront again
						# Reload Those parameter
						tuplePara = self.UpfronRecord_InputSave[ar];
						BaselineStartTime = tuplePara[0]
						SimStartTime = tuplePara[1] 
						SimEndTime = ObserveTime
						Seed_in = tuplePara[4]
						
						command = "./python3 -W ignore ./HawkesSImV2.py" + \
						" Path_Temp=" + TimeStampPath + " Seed=" + str(Seed_in) + " BaselineStartTime=" + str(BaselineStartTime) +\
						" SimStartTime=" + str(SimStartTime) + " SimEndTime=" + str(SimEndTime)+ \
						" PeriodPerPull=" + str(self.PeriodPerPull) + " n_simTime=" + str(1)+\
						" Path_TimeStampTemp=" + TimeStampUpfrontPath + " Path_IntenseUpTemp=" + IntenseUpfrontPath;
						p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
						(output, err) = p.communicate()
						p_status = p.wait()
						
						
						ISMEMOUT = (output.decode('utf8').split("\n")[-2].split(" ")[0] == 'MEMOUT')
						if ISMEMOUT:
							print(  output.decode('utf8').split("\n")[-2] )
							self.Results_WriteOut.write( output.decode('utf8').split("\n")[-2] + "\n");
							exit()
						
						self.UpfronRecord[ar] = ObserveTime;
						#
						self.UpfronRecord_InputSave[ar] = \
						         ( tuplePara[0], tuplePara[1], ObserveTime, tuplePara[3], tuplePara[4] );
					
				tupsIntense = self.Loadtemp(IntenseUpfrontPath)
				if (len(tupsIntense)==1) & (tupsIntense[0]==None):
					UpIntense.append( [0]* self.nsample )
				else:
					temp = [ ins[int( ( ObserveTime - tupsIntense[0] )/self.PeriodPerPull) ] for ins in tupsIntense[2]  ];
					temp = [ [temp[Idx]] * tupsIntense[4][Idx] for Idx in range(0, len(tupsIntense[4]) ) ]
					merged = list(itertools.chain(*temp))
					UpIntense.append( merged )

			#########################################################################################
			Hawkes_mean = [  np.array(up).mean() for up in UpIntense]
			Hawkes_std = [ np.array(up).std() for up in UpIntense]
			#########################################################################################
			UpIntense = [np.array(up).mean() + self.AlphaH*np.array(up).std() for up in UpIntense]
			UpIntense_reshape = np.reshape(UpIntense,( np.int( np.sqrt( self.NumArm ) ), np.int( np.sqrt( self.NumArm ) ) ));
			UpIntense_reshape = gaussian_filter(UpIntense_reshape, self.Lambda )
			UpIntense_smoothout= np.reshape( UpIntense_reshape, (self.NumArm,) );
			#if UpIntense_smoothout.sum()!=0:
			#	UpIntense_smoothout = UpIntense_smoothout / UpIntense_smoothout.sum()
			
			self.Hawkes_Score = UpIntense_smoothout;
			if self.Hawkes_Score[~np.isinf(self.Hawkes_Score)].sum() != 0:
				self.Hawkes_Score_norm = self.Hawkes_Score / self.Hawkes_Score[~np.isinf(self.Hawkes_Score)].sum()
			else:
				self.Hawkes_Score_norm = self.Hawkes_Score
			
			# total score
			#self.PullProb = self.RecrdAvgScrArms + self.Alpha * self.UCB1;
			#self.PullProb = self.PullProb / self.PullProb[~np.isinf(self.PullProb)].sum()
			
			self.ScoreCb = self.query_norm + self.Gamma * self.Hawkes_Score_norm;
			self.queryProb = self.softmax( self.ScoreCb );
			# Fix when num of nonzero prob is less than numPullArm
			if( sum( self.queryProb != 0 ) < self.NumPulledArm):
				IdxNonZero = np.where( self.queryProb!=0 );
				IdxZero = np.where( self.queryProb==0 );
				NumZero = IdxZero[0].shape[0];
				self.queryProb[ IdxZero ] = 10**(-20);
				IdxMax = np.argmax(self.queryProb);
				self.queryProb[IdxMax] = self.queryProb[IdxMax] - 10**(-20) * NumZero;
			
			self.PulledArm = np.random.choice( np.array( range(0, self.NumArm)) , self.NumPulledArm, replace=False, p=self.queryProb.tolist())
			#self.ScoreCb = self.PullProb;
			#self.PulledArm = np.lexsort((np.random.random( self.ScoreCb.shape[0]), self.ScoreCb ))[::-1][ 0: self.NumPulledArm];
			self.ScoreCb = self.ScoreCb[self.PulledArm];
			#
			# Update the Current Index
			self.CurrentIdx_Unb = self.NextIdx_Unb;
			self.CurrentIdx_Ob = self.NextIdx_Ob;

			self.PulledCheck[ self.PulledArm ] = 1;
			self.TimeStmpLast = self.TimeStmp;
			
		self.Results_WriteOut.close()
		
		for ar in range(0, self.NumArm):
			TimeStampPath = self.TimeStampPath+ str(ar) + ".pkl";
			TimeStampUpfrontPath = self.TimeStampUpfrontPath + str(ar) + ".pkl";
			IntenseUpfrontPath = self.IntenseUpfrontPath + str(ar) + ".pkl";
			#
			os.remove( TimeStampPath ); os.remove( TimeStampUpfrontPath ); os.remove( IntenseUpfrontPath );
		####### ####################################
