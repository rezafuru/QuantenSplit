device=${1}
q_depth=${2}
qnn_type=${3}
circuit_type=${4}
q_device=${5}

if [ $# -ne 5 ]; then
  echo "Illegal number of arguments ${#}"
  exit 2
fi

declare -a NO_QUBITS=(2 4 6 8 10 12)
declare -a SEEDS # ENTER SEEDS

export Q_DEPTH="${q_depth}"
export Q_DEVICE=${q_device}

for s in "${SEEDS[@]}"
do
  for q in "${NO_QUBITS[@]}"
  do
    json_str='{"models": {"student_model": {"params": {"circuit_name": "'
    json_str+=${circuit_type}
    json_str+='"}}}}'
    echo ${json_str}
    export QUBITS="${q}";
    .venv/bin/python train_predictors.py --config "config/predictors/FP-baseline_compressor-l032+vessels_felidae_buildings-${qnn_type}.yaml" --device "${device}" --seed "${s}" --result_file "resources/results/$qnn_type/predictors_${circuit_type}_${device}" --json "${json_str}"
  done
done

