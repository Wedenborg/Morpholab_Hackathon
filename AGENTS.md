# Cluster execution

Do not run models, training, inference, evaluation, tests, builds, or other compute
work directly on a login node. Submit these tasks to Slurm with `sbatch`, or use
an allocated compute node through `salloc`/`srun`. Confirm the allocation before
starting work. Login nodes are for lightweight repository inspection, editing,
and job submission/monitoring only. This rule applies to smoke tests as well as
full runs, and to all future agents working in this repository.
