#! /bin/bash

#! /bin/bash

ALGORITHM=$1

INIT_COMMAND_ROOT="srun python HyperSphere/BO/run_BO.py --parallel -g sphere --origin"

case "$ALGORITHM" in
  none) INIT_COMMAND_PREFIX="$INIT_COMMAND_ROOT"
    ;;
  boundary) INIT_COMMAND_PREFIX="$INIT_COMMAND_ROOT --boundary"
    ;;
  warping) INIT_COMMAND_PREFIX="$INIT_COMMAND_ROOT --warping"
    ;;
  warpingboundary) INIT_COMMAND_PREFIX="$INIT_COMMAND_ROOT --warping --boundary"
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
      echo "srun -C cpunode python HyperSphere/BO/run_BO.py -p $EXPPATH"
      eval "srun -C cpunode python HyperSphere/BO/run_BO.py -p $EXPPATH"
    done
  done
done