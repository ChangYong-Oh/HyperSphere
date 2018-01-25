#! /bin/bash

ALGORITHM=$1

INIT_COMMAND_CUBE_ROOT="srun python HyperSphere/BO/run_BO.py --parallel -g cube"
INIT_COMMAND_SPHERE_ROOT="srun python HyperSphere/BO/run_BO.py --parallel -g sphere --origin"

case "$ALGORITHM" in
  cube) INIT_COMMAND_PREFIX="$INIT_COMMAND_CUBE_ROOT"
    ;;
  cubeard) INIT_COMMAND_PREFIX="$INIT_COMMAND_CUBE_ROOT --ard"
    ;;
  none) INIT_COMMAND_PREFIX="$INIT_COMMAND_SPHERE_ROOT"
    ;;
  boundary) INIT_COMMAND_PREFIX="$INIT_COMMAND_SPHERE_ROOT --boundary"
    ;;
  warping) INIT_COMMAND_PREFIX="$INIT_COMMAND_SPHERE_ROOT --warping"
    ;;
  warpingboundary) INIT_COMMAND_PREFIX="$INIT_COMMAND_SPHERE_ROOT --warping --boundary"
    ;;
esac


for DIM in 20 50 100; do
  case "$DIM" in
    20) NEVAL=200
      ;;
    50) NEVAL=400
      ;;
    100) NEVAL=600
      ;;
  esac
  for TASK in branin hartmann6 rosenbrock levy; do
    INIT_COMMAND="$INIT_COMMAND_PREFIX -f $TASK -d $DIM"
    echo "====================COMMAND===================="
    echo "$INIT_COMMAND"
    EVALSTDOUT=$(eval "$INIT_COMMAND")
    echo "$EVALSTDOUT"
    EXPPATH="${EVALSTDOUT##*$'\n'}"
    for ((i=1;i<=$NEVAL;i++)); do
      echo "====================COMMAND===================="
      CONTINUED_COMMAND="srun -C cpunode python HyperSphere/BO/run_BO.py -p $EXPPATH"
      echo "$CONTINUED_COMMAND"
      eval "$CONTINUED_COMMAND"
    done
  done
done
