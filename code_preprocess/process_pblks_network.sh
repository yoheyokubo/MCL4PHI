# assume $SGE_TASK_ID ranges from 1 to 100 (both included)

CODE="process_pblks/process_pblks_network.py"

module load /usr/local/package/modulefiles/python/3.12.0
python3 $CODE $SGE_TASK_ID
