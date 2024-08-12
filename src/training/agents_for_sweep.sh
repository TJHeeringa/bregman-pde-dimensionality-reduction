#!/bin/bash

ARGS=$(getopt -o s:c:t:h --long sweep-long-id:,concurrent-agents:,total-agents:,help,interval -- "$@")

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

HELP=false
SWEEP_ID=
SWEEP_LONG_ID=
CONCURRENT_AGENTS=
TOTAL_AGENTS=
INTERVAL=5
eval set -- "$ARGS"
while true; do
    case "$1" in
        -s | --sweep-long-id ) SWEEP_LONG_ID="$2"; shift 2 ;;
        -c | --concurrent-agents ) CONCURRENT_AGENTS="$2"; shift 2 ;;
        -t | --total-agents ) TOTAL_AGENTS="$2"; shift 2 ;;
        --interval ) INTERVAL="$2"; shift 2 ;;
        -h | --help ) HELP=true; shift ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if $HELP; then
    cat << "EOF"
 This scripts runs Wandb agents on Slurm for your Wandb sweep.

 The following parameters are able to be used:
  -s | --sweep-long-id     The long form of the Wandb sweep id; is of the form <project>/<entity>/<sweep_id>
  -c | --concurrent-agents Maximum number of concurrent Slurm jobs (each with one agent)
  -t | --total-agents      Total number of agents this process will spawn
       --interval          Time interval between checking whether a slurm job is finished
  -h | --help              Raise this help
EOF
    exit 0
fi

total_agents_initiated=0
currently_running_agents=0
runs_completed=0
SWEEP_ID=$(echo "$SWEEP_LONG_ID" | cut -d '/' -f 3)
SCRIPT_DIR=$(dirname "$0")
LOGS_DIR="./logs/$SWEEP_ID"
START_TIME=$(date '+%Y:%m:%d -- %H:%M:%S %Z')

if [ ! -d "$LOGS_DIR" ]; then
  mkdir -p "$LOGS_DIR"
  echo "Logs directory created."
else
  echo "Logs directory already exists."
fi


while [ $runs_completed -lt $TOTAL_AGENTS ];
do
  currently_running_agents=$(squeue --format='%j' | grep -c "$SWEEP_ID")

  runs_completed=$((total_agents_initiated - currently_running_agents))

  echo "
  Sweep: $SWEEP_ID

  Agents:'
    currently:     $currently_running_agents
    initiated:     $total_agents_initiated
    planned:       $TOTAL_AGENTS
    completed:     $runs_completed

  Started at:      $START_TIME
  Last updated at: $(date '+%Y:%m:%d -- %H:%M:%S %Z')

  " > "$LOGS_DIR/sweep.log"
  squeue --name=NotIntendedToExist >> "$LOGS_DIR/sweep.log"
  squeue --format='%.18i %.9P %.30j %.8u %.8T %.10M %.12l %.6D %R' | grep $SWEEP_ID >> "$LOGS_DIR/sweep.log"

  if [ $total_agents_initiated -lt $TOTAL_AGENTS ] && [ $currently_running_agents -lt $CONCURRENT_AGENTS ];
  then
    new_agent_id=$(( total_agents_initiated + 1 ))
    sbatch -o "$LOGS_DIR/job_%j.out" -J "$SWEEP_ID.$new_agent_id" "$SCRIPT_DIR"/start_agent.sb $SWEEP_LONG_ID;
    total_agents_initiated=$new_agent_id
  fi

  sleep $INTERVAL
done
echo "ended"
