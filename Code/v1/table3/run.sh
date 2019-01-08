#!/bin/bash  
> table3/results.txt
printf "#Region Index\t#SAT Assignments\t#CE\tTime(s)\n" >> table3/results.txt
File="table3/results.txt"

#test 1
source table3/test1.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 2
source table3/test2.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 3
source table3/test3.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 4
source table3/test4.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 5
source table3/test5.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 6
source table3/test6.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 7
source table3/test7.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 


#test 8
source table3/test8.sh
printf "A$((FROM_ABS_IDX+1))-R$((FROM_REF_IDX+1))" >> table3/results.txt
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity $VERBOSE --use_ctr_examples $CTR_EX --file $File 