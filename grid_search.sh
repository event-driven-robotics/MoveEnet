#!/usr/bin/env bash


# Default values
script=../train_lenet_decolle.py
MIN_REQ_MEM=5000
save_dir=test
random_search=false
cpu=false
depth=30
max_experiments=5
params_dir=params_to_test

function show_help() {
  echo "Usage: $(basename $0) [OPTIONS]"
  echo "OPTIONS: "
  echo "-s|--script </path-to-training-script>. Default: $script"
  echo "-m|--req_mem <Memory required per simulation in MB>. If available memory is not available in GPU then the script will wait for other experiments to stop befor resuming. Default: $MIN_REQ_MEM"
  echo "-d|--save_dir <Name of subdir to store results>. Default: $save_dir"
  echo "-r|--random_search <Depth of research (Optional)>. Flag to perform random search. Optionally the number of experiments to run can be specified ($depth by default)" 
  echo "--cpu Flag to forcedly run experiments on CPU rather than GPU" 2>&1
  echo "--max_experiments <Max number of experiments to run simultaneously>. Default: $max_experiments"
  echo "--params_dir <Path to diractory containing all the files with the configurations to be tested>. Default: $params_dir"
  exit 1
}

die() {
  printf '%s\n' "$1" >&2
  show_help
}

while :; do
  case $1 in
  -h | -\? | --help)
    show_help # Display a usage synopsis.
    exit
    ;;
  -s | --script) # Takes an option argument; ensure it has been specified.
    if [ "$2" ]; then
      script=$2
      shift
    else
      die "ERROR: option $1 requires a non-empty option argument."
    fi
    ;;
  -m | --req_mem) # Takes an option argument; ensure it has been specified.
    if [ "$2" ]; then
      MIN_REQ_MEM=$2
      shift
    else
      die "ERROR: option $1 requires a non-empty option argument."
    fi
    ;;
  -d | --save_dir) # Takes an option argument; ensure it has been specified.
    if [ "$2" ]; then
      save_dir=$2
      shift
    else
      die "ERROR: option $1 requires a non-empty option argument."
    fi
    ;;
    --params_dir) # Takes an option argument; ensure it has been specified.
    if [ "$2" ]; then
      params_dir=$2
      shift
    else
      die "ERROR: option $1 requires a non-empty option argument."
    fi
    ;;
  -r | --random_search) # Takes an option argument; ensure it has been specified.
    if [ "$2" ]; then
      depth=$2
      shift
    fi
    random_search=true
    ;;  
    --max_experiments) # Takes an option argument; ensure it has been specified.
    if [ "$2" ]; then
      max_experiments=$2
      shift
    fi
    ;;  
    --cpu) # Takes an option argument; ensure it has been specified.
    cpu=true
    ;;
  --) # End of all options.
    shift
    break
    ;;
  -?*)
    printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
    ;;
  *) # Default case: No more options, so break out of the loop.
    break ;;
  esac

  shift
done

#Kill all child processes on exit or ctrl-c
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

[[ ! -d "$params_dir" ]] && die "Cannot find $params_dir directory"

if [[ "$random_search" == true ]]; then
  echo "Running random search on a subset of $depth parameter settings."
else
  echo "Running grid search on all parameter settings."
fi
echo "Training script: $script"
echo "Memory required per simulation: $MIN_REQ_MEM"
echo "Saving results in subdir $save_dir"
echo


#Check if and how many GPUs are available
if [[ "$cpu" == false ]] && hash nvidia-smi 2>/dev/null; then
  NUM_GPUS=$(nvidia-smi --query-gpu=count --format="csv,noheader" | head -n 1)
  echo "Found ${NUM_GPUS} GPUs."
else
  NUM_GPUS=0
  echo "Running on CPU"
fi

param_files=($params_dir/*)
if [ "$random_search" = true ]; then
  param_files=($(shuf -e "${param_files[@]}"))
  param_files=("${param_files[@]:0:depth}")
fi

num_running_experiments=0
for filename in "${param_files[@]}"; do
  
  if [ "${NUM_GPUS}" -gt "0" ]; then

    # Look for available GPU
    while :; do
      MAX_FREE_MEM=0

      for ((j = 0; j < $NUM_GPUS; j++)); do
        echo "Checking availability on GPU $j"
        MEM=0
        for i in {1..10}; do
          ((MEM += $(nvidia-smi --query-gpu=memory.free --format="csv,noheader,nounits" --id=${j})))
                              sleep 1
        done
        ((MEM /= 10))
        if [ "$MEM" -gt "$MAX_FREE_MEM" ]; then
          max_j=$j
          MAX_FREE_MEM=$MEM
        fi
      done

      echo "The GPU with most free memory is GPU $max_j with $MAX_FREE_MEM MB"
      if [ "$MAX_FREE_MEM" -gt "$MIN_REQ_MEM" ]; then # If more than 10GB are free the n use that GPU
        break
      else
        echo "You need more than $MIN_REQ_MEM MB to launch the simulation."
      fi

      echo "[$(date)]All GPUS are busy. Waiting..."
      wait -n
    done

    # Running one simulation
    echo "[$(date)]Running on GPU number ${max_j} with parameter file ${filename}. Please check logs dir for execution logs"
    [ ! -d logs/"$save_dir" ] && mkdir -p logs/"$save_dir"
    curr_save_dir=$save_dir/`date +"%Y_%m_%d_%H_%M_%S"`
    mkdir $curr_save_dir
    CUDA_VISIBLE_DEVICES="${max_j}" python -u "$script" --params_file "${filename}" --save_dir "$curr_save_dir" &> $curr_save_dir/log.txt &
    sleep 60

  else
    if [ "${num_running_experiments}" -lt "${max_experiments}" ]; then
      echo "[$(date)]Running on CPU with parameter file ${filename}. Please check logs dir for execution logs"
      curr_save_dir=$save_dir/`date +"%Y_%m_%d_%H_%M_%S"`
      mkdir $curr_save_dir
      CUDA_VISIBLE_DEVICES='' python -u "$script" --params_file "${filename}" --save_dir "$curr_save_dir" &
      sleep 1
    else
      echo "Reached the maximum number of $num_running_experiments experiments running simultaneously. Waiting..."
      wait -n
    fi
    num_running_experiments=`pgrep -c -P$$ -f python`
    echo "Currently running experiments: $num_running_experiments"
  fi
done

echo "No more simulations to launch. Waiting for the ones currently running to finish"
wait # For the last simulations to finish
echo "All processes have finished their job. Exiting"

exit 0