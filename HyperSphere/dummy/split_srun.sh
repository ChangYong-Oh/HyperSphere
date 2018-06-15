#!/bin/bash

run_command="srun python ./HyperSphere/BO/run_BO.py"

case $1 in
'cube')
    exp_config="-g cube";;
'cubeard')
    exp_config="-g cube --ard";;
'sphere')
    exp_config="-g sphere --origin";;
'sphereboundary')
    exp_config="-g sphere --origin --boundary";;
'spherewarping')
    exp_config="-g sphere --origin --warping";;
'spherewarpingboundary')
    exp_config="-g sphere --origin --warping --boundary";;
esac

case $2 in
20)
    n_eval=200;;
50)
    n_eval=400;;
100)
    n_eval=600;;
200)
    n_eval=800;;
1000)
    n_eval=1000;;
esac

for s in {1..5}
do
    dir_name=$($run_command $exp_config -d $2 -f $3 | tail -1)
    for (( i=1; i<=$n_eval; i++ ))
    do
        $run_command -p $dir_name
    done
done