#!/bin/bash  
> table2/results.txt
echo -e "#Hidden Layers\t#Neurons\t#SAT Assignments\t#CE\tTime(s)\n" >> table2/results.txt
File="table2/results.txt"

#test 1
source table2/test1.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 2
source table2/test2.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 3
source table2/test3.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 4
source table2/test4.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 5
source table2/test5.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 6
source table2/test6.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 7
source table2/test7.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 8
source table2/test8.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 9
source table2/test9.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 10
source table2/test10.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 

#test 11
source table2/test11.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 12
source table2/test12.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 13
source table2/test13.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 14
source table2/test14.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 15
source table2/test15.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 

# 2 Hidden layers networks

#test 16
source table2/test16.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 17
source table2/test17.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 18
source table2/test18.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 19
source table2/test19.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 20
source table2/test20.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 21
source table2/test21.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 22
source table2/test22.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 23
source table2/test23.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 



#test 24
source table2/test24.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


3 Hidden layers networks

test 25
source table2/test25.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 

#test 26
source table2/test26.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 

#test 27
source table2/test27.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 

4 Hidden layers networks

test 28
source table2/test28.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 

#test 29
source table2/test29.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 

