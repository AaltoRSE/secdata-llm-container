# secdata-llm-container build and test

## Files
- `sec_llm.def` - container definition file
- `env.yml` - environment file
- `build.sh` - build script
- `container-scripts/` - runtime helper scripts copied into the image
- `container-scripts/start_all.sh` - start common services (convenience wrapper)
- `container-scripts/start_ollama.sh` - start Ollama server inside the container
- `container-scripts/start_code_server.sh` - start code-server inside the container
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

### Conda environments by model

The container includes three prebuilt conda environments:

- `sec-llm-env` (default; auto-activated)
- `sec-llm-env2`
- `sec-llm-env3`

Use different environments for different models:

| model ID | Conda env to use |
|-------------------|------------------|
| `utter-project/EuroLLM-9B-Instruct` | `sec-llm-env2` |
| `intfloat/multilingual-e5-large-instruct` | `sec-llm-env` |
| `TurkuNLP/gpt3-finnish-8B` | `sec-llm-env` |
| `Helsinki-NLP/opus-mt-fi-en` | `sec-llm-env` |
| `Helsinki-NLP/opus-mt-en-fi` | `sec-llm-env` |
| `facebook/nllb-200-3.3B` | `sec-llm-env` |
| `google/gemma-4-E4B-it` | `sec-llm-env3` |

Example (inside container):

```bash
# default env for most models
python tests/teste5.py

# explicit env for EuroLLM
/opt/conda/envs/sec-llm-env2/bin/python tests/eurollm.py

# explicit env for Gemma
/opt/conda/envs/sec-llm-env3/bin/python tests/gemma.py
```

## Runtime services (JupyterLab, code-server, Ollama)

These are meant for use on a **VDI desktop** (secdata, e.g.) with **no internet**: bind services to `127.0.0.1` and open them in a browser on the same machine.

| Service        | Script                         | Default URL              |
|----------------|--------------------------------|--------------------------|
| Ollama API     | `/opt/dev/start_ollama.sh`     | `http://127.0.0.1:11434` |
| JupyterLab     | `/opt/dev/start_jupyter_lab.sh`| `http://127.0.0.1:8888`  |
| code-server    | `/opt/dev/start_code_server.sh`| `http://127.0.0.1:9090`  |

**Persist Ollama models** (recommended): create a directory on the host and bind-mount it to `/ollama_models` so weights are not lost when the container exits. Set `OLLAMA_HOST_DIR` to that path—replace the placeholder below.

```bash
OLLAMA_HOST_DIR="/path/to/ollama_models"
mkdir -p "$OLLAMA_HOST_DIR"
singularity shell --bind "$OLLAMA_HOST_DIR:/ollama_models" sec_llm.sif
```

Inside the container, start the services you need. Most users run **either** JupyterLab **or** code-server (VS Code in the browser).

**Ollama is optional for code-server**: you can use code-server by itself for editing/running code. Only start Ollama if you want a **local LLM API** (e.g., for the Continue extension).

If you do want LLM features, **Ollama and code-server can run at the same time** in one container—same ports as in the table above (`11434` and `9090`).

```bash
# Primary UI (pick one):
/opt/dev/start_jupyter_lab.sh
/opt/dev/start_code_server.sh
# Optional (local LLM API for Continue; see below for how to run alongside code-server):
/opt/dev/start_ollama.sh
```

**Ollama + code-server together**

Use **two shells in the same Apptainer instance**: start an **instance** once on the host, open two terminals, and `shell` into the **same** instance name in each—then run `/opt/dev/start_ollama.sh` in one terminal and `/opt/dev/start_code_server.sh` in the other. Plain `singularity shell sec_llm.sif` / `apptainer shell sec_llm.sif` in two windows is usually **two separate container runs**; instance mode is what ties both terminals to one container.

On the host (adjust image path, `OLLAMA_HOST_DIR`, and `--bind` to match your setup; `singularity` and `apptainer` are equivalent on many systems):

```bash
OLLAMA_HOST_DIR="/path/to/ollama_models"
mkdir -p "$OLLAMA_HOST_DIR"
apptainer instance start --bind "$OLLAMA_HOST_DIR:/ollama_models" sec_llm.sif llm-dev
```

Terminal A:

```bash
apptainer shell instance://llm-dev
/opt/dev/start_ollama.sh
```

Terminal B:

```bash
apptainer shell instance://llm-dev
/opt/dev/start_code_server.sh
```

When finished, from the host:

```bash
apptainer instance stop llm-dev
```

You may also need to bind-mount additional host directories (for example, datasets, project repos, model files, or caches) depending on your workflow. Use the same binds on `instance start` that you would use for a normal `shell`/`exec`.

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

