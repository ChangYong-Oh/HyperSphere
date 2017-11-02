#! /bin/bash

REP=$1
for ((i=1;i<=REP;i++)); do
  eval 'srun' ${@:2}
  sleep 3
done
