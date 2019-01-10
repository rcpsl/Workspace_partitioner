#!/bin/bash  
> table4/results.txt
printf "#Hidden Layers\t#of Neurons\t   Time(s) using CE\tTime(s) without CE\n" >> table4/results.txt
File="table4/results.txt"

#test 1
source table4/test1_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test1_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test1.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File


#test 2
source table4/test2_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test2_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test2.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 3
source table4/test3_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test3_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test3.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 4
source table4/test4_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test4_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test4.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 5
source table4/test5_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test5_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test5.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File


#### 2 Hidden layers networks

#test 6
source table4/test6_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test6_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test6.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 7
source table4/test7_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test7_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test7.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 8
source table4/test8_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test8_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test8.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 9
source table4/test9_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test9_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test9.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 10
source table4/test10_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test10_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test10.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 11
source table4/test11_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test11_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test11.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 12
source table4/test2_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test2_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test2.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 13
source table4/test13_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test13_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test13.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 14
source table4/test14_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test14_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test14.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#### 3 Hidden layers network
#test 15
source table4/test15_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test15_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test15.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 16
source table4/test16_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test16_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test16.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File

#test 17
source table4/test17_preprocess.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX
source table4/test17_CE.sh
printf "$((NLAYERS-1))\t\t$(((NLAYERS-1)*H +2))" >> $File
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
source table4/test17.sh
python2.7 state_machine.py $NLAYERS $H $FROM_ABS_IDX $FROM_REF_IDX $TO_ABS_IDX $PREPROCESS --verbosity OFF --use_ctr_examples $CTR_EX --file $File 
echo -e "\n------------------------------------------------------------------------" >> $File


