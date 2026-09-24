import os
#
# This is a user-modifiable Python file designed to be a set of simple input file and directory settings that you can choose and change.

# The location of the file containing the AssetCollection id for the dtk sif (Singularity image)
sif_id = 'dtk_sif.id'

# The script is going to use this to store the downloaded schema file. Create 'download' directory or change to your preferred (existing) location.
schema_file=os.path.join(os.path.expanduser('~'),"download/schema.json")

# The script is going to use this to store the downloaded Eradication binary. Create 'download' directory or change to your preferred (existing) location.
eradication_path=os.path.join(os.path.expanduser('~'),"download/Eradication")

# Create 'Assets' directory or change to a path you prefer. idmtools will upload files found here.
assets_input_dir="Assets"
plugins_folder = "download/reporter_plugins"
# This is where your inputs are located
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
input_dir = os.path.join(ROOT_DIR, "inputs")
Output_dir = os.path.join(ROOT_DIR, "outputs")


ep4_path="python_scripts"
#requirements = "./requirements.txt"
# This is where your simulations and outputs will be stored
user = os.getlogin()
job_directory = os.path.join(os.path.expanduser('~'), 'Final-Project-UWR/experiments')
os.makedirs(job_directory, exist_ok=True)
# This is the path to the sif image used to run EMOD
# dtk_run_rocky_py39.sif ships Python 3.9 on an old Rocky base whose glibc is too old for the
# pip-installed Eradication binary (built for Amazon Linux 2023). Using a custom-built image with
# a matching AL2023 base instead - see containers/dtk_run_al2023.def.
SIF_PATH = os.path.join(ROOT_DIR, "containers/dtk_run_al2023.sif")

# ---------------------------------------------------------------------------
# SLURM / platform settings (AWS)
# These are the only environment-specific settings most users need to change.
# On a different cluster, edit the values below - you should not need to touch
# run_example.py.
# ---------------------------------------------------------------------------

# SLURM partition (queue) to submit jobs to.
partition = 'demo'

# Environment module(s) that provide singularity/apptainer on this cluster.
# On AWS this is a filesystem path to the modulefile; on other clusters it may
# be a module name like 'singularity/3.8.1'. idmtools passes these to `module load`.
singularity_module = '/shared/emod/shared_tools/modulefiles/singularity'

# Wall-clock time limit requested per job (HH:MM:SS).
sim_time = '10:00:00'

# Maximum number of simulations SLURM will run at once for one experiment.
max_running_jobs = 100
#max_running_jobs = 5