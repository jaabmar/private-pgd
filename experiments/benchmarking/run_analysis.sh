# call the script run_analysis.sh project_name\
# project_name is the name of the project folder in experiments/analysis/table
# e.g. run_analysis.sh benchmark_table
source experiments/run_setup.sh
project_name="benchmark_table"
echo $project_name

# Run the Python post-processing and sbatch commands
sbatch --time=3:59:59 --tmp=1024 --gpus=1 --mem-per-cpu=10g --gres=gpumem:10g  --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics.py --project_name $project_name --metrics wdist_1_2Way"
sbatch --time=23:59:59 --tmp=1024 --cpus-per-task=10 --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_downstream.py --project_name $project_name"
sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics.py --project_name $project_name --metrics cov_fixed"
sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics.py --project_name $project_name --metrics newl1_2Way"
sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics.py --project_name $project_name --metrics rand_thresholding_query"
sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics.py --project_name $project_name --metrics rand_counting_query"


#python experiments/post_processing/eval_statistics.py --project_name benchmark_table --metrics wdist_1_2Way"
# sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics.py --project_name benchmark_table --metrics rand_thresholding_query"
# sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics.py --project_name benchmark_table --metrics rand_counting_query"


sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics_bench.py --project_name rap   --metrics rand_thresholding_query"
sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics_bench.py --project_name  rap  --metrics rand_counting_query"

sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics_bench.py --project_name  private_gsd  --metrics rand_thresholding_query"
sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics_bench.py --project_name  private_gsd --metrics rand_counting_query"

sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics_bench.py --project_name gem  --metrics rand_thresholding_query"
sbatch --time=3:59:59 --tmp=1024  --mem-per-cpu=10g --account=es_yang --wrap="srun python experiments/post_processing/eval_statistics_bench.py --project_name  gem  --metrics rand_counting_query"

#srun --pty --time=3:59:59 --tmp=1024 --gpus=1 --mem-per-cpu=10g --gres=gpumem:10g --account=es_yang /bin/bash
