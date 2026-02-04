# secdata-llm-container build and test

## Files
- `sec_llm.def` - container definition file
- `env.yml` - environment file
- `build.sh` - build script
- `tests.sh` - test script
- `tests/*.py` - python scripts for testing

## Build
```
sbatch build.sh
```
A singularity container will be built and saved as `sec_llm.sif` (this is the file you need to copy to secdata, around 20GB)

## Tests
```
sbatch tests.sh
```

## Start a jupyter lab session
```
# start a shell in the container
singularity shell sec_llm.sif

# start jupyter lab
jupyter lab 
```

## Label Studio

Label Studio is included in the container for data annotation tasks. To run it:

1. **Create a data directory** (for persisting projects and annotations):
   ```
   mkdir -p label_studio_data
   ```

2. **Start Label Studio** (bind-mount the data directory so annotations are saved on the host):
   ```
   singularity exec --bind ./label_studio_data:/label_studio_data sec_llm.sif /opt/label-studio/start_label_studio.sh
   ```

3. **Open the web UI** at http://localhost:8080 in your browser.

4. **First-time setup**: On first launch, create an account. Projects and annotations are stored in `label_studio_data/` on your host.

**Note**: If running on a remote machine, use SSH port forwarding to access the UI:
   ```
   ssh -L 8080:localhost:8080 user@remote-host
   ```
   Then start Label Studio on the remote host and open http://localhost:8080 locally.

