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
A singularity container will be built and saved as `sec_llm.sif` (this is the file you need to copy to secdata, around 20GB). 

## Audience-specific guide

### For container developers (build machine with internet)

1. Use an up-to-date checkout (`sec_llm.def`, `env.yml`, `container-scripts/`, `nltk_data/` as referenced in the def).
2. **Build** (needs network for `%post` and conda): `sbatch build.sh` or your usual `apptainer build` command.
3. **Optional:** `sbatch tests.sh` to sanity-check the image.
4. **Ollama weights are separate from the `.sif`:** the image contains the Ollama *binary*, not model blobs. You can either:
   - **Pull tags online** (`ollama pull ...`) on a host with internet, or
   - **Prepare GGUF + Modelfile assets** for offline `ollama create` on the target host.
5. Share this README and `sec_llm.sif` with end users.

Optional online preload flow:
   ```bash
   mkdir -p ./ollama_models
   export OLLAMA_MODELS="$PWD/ollama_models"
   ollama pull llama3.2   # or whatever tag you want
   ```
Copy the whole **`ollama_models`** directory to secdata (it can be many GB).

### For container users (secdata, offline)

#### What to copy (besides the `.sif`)

| Bring to secdata | Why |
|--------------|-----|
| **`sec_llm.sif`** | The container image (admin transfer is fine). Everything baked in at build time (Python env, Jupyter, code-server, Continue, Ollama binary, NLTK data from build) is inside this file. |
| **`ollama_models/`** (directory) | **Required for local LLMs** if you did not put models inside the image. Bind-mount to `/ollama_models` when you run Singularity. |
| **Your project folders** | Code, notebooks, datasets, configs—normal workspace trees you bind-mount or work from on secdata. |
| **`huggingface-hub` cache** (optional) | If you run Transformers/vLLM with `HF_HOME=/models/huggingface-hub`, create that cache on a networked machine and copy it; bind-mount e.g. `--bind /path/on/secdata/hf_cache:/models/huggingface-hub`. |
| **`label_studio_data/`** (optional) | Only if you use Label Studio and want existing projects/annotations on secdata. |


## Tests
```
sbatch tests.sh
```

## Runtime services (JupyterLab, code-server, Ollama)

These are meant for use on a **VDI desktop** (secdata, e.g.) with **no internet**: bind services to `127.0.0.1` and open them in a browser on the same machine.

| Service        | Script                         | Default URL              |
|----------------|--------------------------------|--------------------------|
| Ollama API     | `/opt/dev/start_ollama.sh`     | `http://127.0.0.1:11434` |
| JupyterLab     | `/opt/dev/start_jupyter_lab.sh`| `http://127.0.0.1:8888`  |
| code-server    | `/opt/dev/start_code_server.sh`| `http://127.0.0.1:9090`  |

**Persist Ollama models** (recommended): create a directory on the host and bind-mount it to `/ollama_models` so weights are not lost when the container exits.

```bash
mkdir -p "$HOME/ollama_models"
singularity shell --bind "$HOME/ollama_models:/ollama_models" sec_llm.sif
```

Inside the container, start the services you need. Most users run **either** JupyterLab **or** VSCode/code-server. Ollama + Continue are optional and mainly for the VSCode/code-server workflow:

```bash
# Choose one primary interface:
/opt/dev/start_jupyter_lab.sh
/opt/dev/start_code_server.sh 
# Optional (for local LLM serving and Continue in code-server):
/opt/dev/start_ollama.sh
```

You may also need to bind-mount additional host directories (for example, datasets, project repos, model files, or caches) depending on your workflow.

Environment overrides: `JUPYTER_PORT`, `CODE_SERVER_PORT`, `CODE_SERVER_AUTH` (`none` or `password`; with `password`, set `PASSWORD`), `OLLAMA_HOST`, `OLLAMA_MODELS`.

#### Ollama model setup options for end users

Use one of these options depending on what you can transfer to the remote host.

**Option A (online preload elsewhere, then copy `ollama_models/`)**: on any machine **with** internet, pull models into a directory, then copy that directory to secdata (via admin transfer) and bind it to `/ollama_models`:

```bash
export OLLAMA_MODELS=/path/to/ollama_models
ollama pull llama3.2
# Copy /path/to/ollama_models to secdata, bind as above, then `ollama list` / chat in Continue.
```

**Option B (offline host, transfer GGUF files only)**: if you can only copy model files to the remote server, create local Ollama models from GGUF directly on the remote host.

On the remote host, prepare a folder with your `.gguf` files and a `Modelfile`:

```bash
mkdir -p "$HOME/gguf/my-model"
cp /path/to/model.gguf "$HOME/gguf/my-model/model.gguf"
cat > "$HOME/gguf/my-model/Modelfile" <<'EOF'
FROM ./model.gguf
EOF
```

Then, inside the running container (with Ollama server started), create and verify the model:

```bash
export OLLAMA_MODELS=/ollama_models
cd "$HOME/gguf/my-model"
ollama create mymodel -f Modelfile
ollama list
```

Use `mymodel` in Continue config (`~/.continue/config.yaml`) or via Ollama CLI.

After your first code-server start, **`~/.continue/config.yaml`** is created from the template. You can **change the model anytime** by editing that file or using Continue’s settings UI; use the same name as in `ollama list`.

**Continue** is pre-installed under `/opt/code-server/extensions`. In code-server, sign in / marketplace calls will fail offline; use the bundled extension only unless you sideload more `.vsix` files at build time.

## Quick start (minimal)

```bash
singularity shell sec_llm.sif
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

