#!/bin/bash 
H="20"  #Number of Neurons per hidden layer
FROM_ABS_IDX="1"     #Starting Region abstract index
FROM_REF_IDX="2" #Starting Regions refined index
TO_ABS_IDX="2"       #End Region abstract index
TO_REF_IDX="3"   #End Regions refined index
PREPROCESS="0"              #Generate counter examples
CTR_EX="1"    #Use Counter examples for the feasability problem, used when preprocess is 0 only
NLAYERS="4"                 #Number of hidden layers + output layer
VERBOSE="ON"              