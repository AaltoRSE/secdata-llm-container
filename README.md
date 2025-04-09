# secdata-llm-container build and test

## Files
- `sec_llm.def` - container definition file
- `env.yml` - environment file
- `build.sh` - build script
- `tests.sh` - test script
- `*.py` - python scripts for testing

## Build
```
sbatch build.sh
```
A singularity container will be built and saved as `sec_llm.sif` (this is the file you need to copy to secdata, around 15GB)

## Tests
```
sbatch tests.sh
```

## Start a jupyter lab session
```
# start a shell in the container
singularity shell sec_llm.sif

# activate the conda environment
source activate sec-llm-env

# start jupyter lab
jupyter lab 
```
