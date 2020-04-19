###### import modules BEGIN ######
import numpy as np                              # Array
import pdb                                      # Debugging
###### import modules END   ######

'''
Description:
  Unobserved Data is considered as Ground Truth (311 call, Drug OD record, etc.)
  Observed Data is considered as Side information (Twitter, OD ambulance, etc.)

  Both of them should be in the following format (Tab deliminated):
        Y_Latitude X_Longitude Time_Stamp Arm_Index 

  Variable Name for the path of Unobserved Data: UnbDataPath
  Variable Name for the path of Observed Data: ObDataPath
  Variable Name for the path of Feature of Unobserved Data: UnbFeatPath
  Variable Name for the path of Feature of Observed Data: ObFeatPath
'''


def DataInput(UnbDataPath):
	DictPara = {}; # dictionary for parametization

	###### read the whole content  ######
	with open(UnbDataPath, 'r') as f: # Unobserved Data
		contentUnb = f.read();
	f.close();
	
	#with open(ObDataPath, 'r') as f: # Observed Dat
	#	contentOb = f.read();
	#f.close();
	######################################

	###### break down the content  ######
	contentUnb = contentUnb.split("\n")[:-1];
	#contentOb = contentOb.split("\n")[:-1];
	#####################################

	'''
	########## get the first line #######
	Par_Str = (contentUnb[0]).split(' ');
	if '' in Par_Str:
		Par_Str.remove('');
	Par_Str = Par_Str + (contentOb[0]).split(' ');
	if '' in Par_Str:
		Par_Str.remove('');
	######################################

	###### Paramertization the first line ######

	for paraIdx in range(1, len( Par_Str ) ):

		para = Par_Str[ paraIdx ];
		para = para.split("=");

		# Since in Unb and Ob, the parameter are the same...
		if para[0] not in DictPara:	
			DictPara[para[0]] = para[1];

	###### ############################# ######
	'''
	###### Read Y X Time Arm data ######
	UnbData = np.float_( [ (contentUnb[i].split('\t'))[0:4] for i in range(0,len(contentUnb)) ] );
	#ObData  = np.float_( [ (contentOb[i].split('\t'))[0:4] for i in range(0,len(contentOb)) ] );
		
	#UnbData_Feature = [ (contentUnb[i].split('\t'))[4:] for i in range(1,len(contentUnb)) ];
	#ObData_Feature = [ (contentOb[i].split('\t'))[4:] for i in range(1,len(contentOb)) ];
	###### ###################### ######

	DictPara['UnbData'] = UnbData;
	#DictPara['ObData'] = ObData;
	#DictPara['UnbData_Feature'] = UnbData_Feature;
	#DictPara['ObData_Feature'] = ObData_Feature;

	print(DictPara.keys())
	return DictPara;

