###### import modules BEGIN ######

import sys                      # Get parameter from command line

###### import modules END   ######

import pdb
def ComdPara(ArgvStr):
	DictPara = {};
	for paraIdx in range(1, len( ArgvStr ) ):
		para = sys.argv[ paraIdx ];
		#print("Command Line: ", para);
		para = para.split("=");
		DictPara[para[0]] = para[1];
	print( DictPara.keys() )
	return DictPara;
