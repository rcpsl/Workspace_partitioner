#!/bin/bash 
cd region_partition 
File="../table1/results.txt"
> $File
printf "# of vertices\t  # of lasers\t# of Regions\tTime(s)\n" >> $File

#test 1
source ../table1/test1.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 2
source ../table1/test2.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 3
source ../table1/test3.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 4
source ../table1/test4.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 5
source ../table1/test5.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 6
source ../table1/test6.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 7
source ../table1/test7.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 8
source ../table1/test8.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 9
source ../table1/test9.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 10
source ../table1/test10.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

#test 11
source ../table1/test11.sh
printf "    $((NUM_VERTICES))\t\t\t$((NUM_LASERS))" >> $File
python2.7 state_machine.py $NUM_VERTICES $NUM_LASERS --file $File
echo -e "\n--------------------------------------------------------">> $File

cd -