
#!/bin/bash

# Export relevant paths
source experiments/run_setup.sh
# Define the directory containing your folders
# Folder keywords
declare -a prefixes=("*ans*" "*medical*" "*nyc*" "*Diabetes*"  "*black_friday*")


declare -a disc_types=("default") 
declare -a folder_keywords
declare -a suffixes=("32")
declare -A hyperparams

regularized=false
hyperparams[max_cells]="100000"
hyperparams[n_particles]="100000"
hyperparams[epsilon]="0.2 1.0 2.5"
hyperparams[run_number]="1 2 3 4 5"
PROJECT_NAME="experiment_marginals"

#!/bin/bash






constraints() {
    hyperparams[inference_type]="advanced_extended"
    hyperparams[penalize_reg]="1"
    hyperparams[mechanism]="KWay"
    regularized=true
    hyperparams[scale_reg]="0.0 0.01 0.1 1.0 10.0 100.0"
    prefixes=("*ans_inc*" "ans_emp*")
    hyperparams[epsilon]="2.0"
    hyperparams[epsilon_reg]="0.5"
    hyperparams[run_number]="1 2 3 4 5"
    hyperparams[delta]="0.000008"
    PROJECT_NAME="experiment_constraint_new"
    hyperparams[workload_type]="Base"
    hyperparams[regularizer_type]="thresquery_1"
    hyperparams[proper_normalization]="1"

}


table1() {
    suffixes=("_32")

    PROJECT_NAME="benchmark_table"
    prefixes=("*ans*" "*medical*" "*nyc*" "*Diabetes*"  "*black_friday*")
    hyperparams[inference_type]="privpgd"
    hyperparams[mechanism]="KWay"
    hyperparams[p_mask]="20"


    hyperparams[scheduler_step]="50"
    hyperparams[scheduler_gamma]="0.75"


    hyperparams[scheduler_step_proj]="200"
    hyperparams[iters_proj]="2000"
    hyperparams[scheduler_gamma_proj]="0.75"
    hyperparams[num_projections_proj]="20"
    hyperparams[degree]="2"
    hyperparams[lr]="0.1"
    hyperparams[iters]="1000"
    hyperparams[n_particles]="500000"

    SBATCH_ARGS=" --time=3:59:59 --tmp=1024 --gpus=1 --mem-per-cpu=10g --gres=gpumem:10g --account=$ACCOUNT_NAME"
}
table2() {
    suffixes=("_32")

    PROJECT_NAME="benchmark_table"
    prefixes=("*")

    hyperparams[inference_type]="pgm_euclid"
    hyperparams[mechanism]="MST"
    hyperparams[degree]="2"
    hyperparams[iters]="3000"
    hyperparams[max_model_size]="80"
    hyperparams[lr]="1.0"

    SBATCH_ARGS=" --time=23:59:59 --tmp=1024 --mem-per-cpu=10g --account=$ACCOUNT_NAME"
}

table3() {
    suffixes=("_32")

    PROJECT_NAME="benchmark_table"
    prefixes=("*")

    hyperparams[inference_type]="pgm_euclid"
    hyperparams[mechanism]="AIM"
    hyperparams[degree]="2"
    hyperparams[iters]="3000"
    hyperparams[max_model_size]="80"
    hyperparams[lr]="1.0"

    SBATCH_ARGS=" --time=23:59:59 --tmp=1024 --mem-per-cpu=10g --account=$ACCOUNT_NAME"
}







table4() {
    PROJECT_NAME="experiment_table_marginals"
    prefixes=("*")
    hyperparams[inference_type]="advanced_extended localPGM"
    #hyperparams[inference_type]="pgm_euclid advanced_sliced"
    hyperparams[mechanism]="KWay"
    hyperparams[epsilon]="0.2 1.0 2.5"
    hyperparams[descent_type]="MD"
    hyperparams[workload_type]="Base"
    hyperparams[degree]="2"
    suffixes=("_32")
    hyperparams[proper_normalization]="1"
    hyperparams[only_marginals]="1"

}

hyperparams() {
    suffixes=("_32")
    hyperparams[run_number]="1 2"
    PROJECT_NAME="hyperparams2"
    prefixes=("*")

    hyperparams[inference_type]="privpgd"
    hyperparams[mechanism]="KWay"
    hyperparams[degree]="2"
    hyperparams[lr]="0.1"
    hyperparams[iters]="1000"
    hyperparams[p_mask]="20"
    hyperparams[scheduler_step]="50"
    hyperparams[scheduler_gamma]="0.75"
    hyperparams[epsilon]="2.5"


    hyperparams[scheduler_step_proj]="200"
    hyperparams[iters_proj]="2000"
    hyperparams[scheduler_gamma_proj]="0.75"
    hyperparams[num_projections_proj]="20 100"

    SBATCH_ARGS=" --time=3:59:59 --tmp=1024 --gpus=1 --mem-per-cpu=10g --gres=gpumem:10g --account=$ACCOUNT_NAME"
}













main() {
    local experiment_type=$1

    case "$experiment_type" in
    "constraints")
        constraints ;;
    "table1")
        table1 ;;
    "table2")
        table2 ;;
    "table3")
        table3 ;;
    "hyperparams")
        hyperparams ;;
    *)
        echo "Unknown experiment type: $experiment_type"
        exit 1
        ;;
    esac


    for prefix in "${prefixes[@]}"; do
        for disc_type in "${disc_types[@]}"; do
            for suffix in "${suffixes[@]}"; do
                folder_keywords+=("${prefix}${disc_type}*${suffix}")
            done
        done
    done

    echo "${folder_keywords[@]}"

    rm tmp_args.txt
    touch tmp_args.txt

    generate_combinations() {
        local keys=("${!hyperparams[@]}")
        generate_combinations_recursive "${keys[@]}"
    }

    generate_combinations_recursive() {
        local key="$1"
        shift
        for value in ${hyperparams[$key]}; do
            if [ "$#" -gt "0" ]; then
                generate_combinations_recursive "$@" | while IFS= read -r line; do
                    echo "--$key $value $line"
                done
            else
                echo "--$key $value"
            fi
        done
    }

    # Initialize a string to accumulate the combinations
    all_combinations=""

    # Collect all the combinations
    for keyword in "${folder_keywords[@]}"; do
        for folder in $BASE_DIR/*$keyword*; do
            if [ -d "$folder" ]; then
                DATASET_NAME=$(basename "$folder")
                while IFS= read -r cmd_args; do
                    parameters=""
                    for param in $cmd_args; do
                        # Convert "--key value" to "key=value"
                        if [[ $param == "--"* ]]; then
                            param_key=${param#"--"}
                            parameters+="$param_key="
                        else
                            param_value=$param
                            parameters+="$param_value "
                        fi
                    done
                    #all_combinations+="$DATASET_NAME:$parameters,"
                    echo "$DATASET_NAME:$parameters" >> tmp_args.txt

                done < <(generate_combinations)
            fi
        done
    done


    output=$(python experiments/check_wandb.py $PROJECT_NAME tmp_args.txt)

    echo $output

    IFS=',' read -ra results <<< "$output"
    index=0

    for keyword in "${folder_keywords[@]}"; do
        for folder in $BASE_DIR/*$keyword*; do
            if [ -d "$folder" ]; then
                DATASET_NAME=$(basename "$folder")
                while IFS= read -r cmd_args; do
                    if [ "${results[$index]}" -eq "0" ]; then
                        sbatch $SBATCH_ARGS \
                            --job-name="private_ot_${DATASET_NAME}_gan_params" \
                            --output=$SLURM_TMPDIR"/private_ot_${DATASET_NAME}_gan_params"_%j.txt \
                            --wrap="srun python experiments/main.py --project_name $PROJECT_NAME --regularized $regularized --base_path $folder $cmd_args "
                        
                    fi
                    ((index++))
                done < <(generate_combinations)
            fi
        done
    done





    # The rest of your code which uses hyperparams, runs combinations, etc.
}

# Entry Point
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_type>"
    exit 1
fi

main "$1"



