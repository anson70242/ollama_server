## A Script to use ollama server from anywhere
### Note

Please set up your development environment and execute all code on your local machine. The Large Language Model (LLM) will be the only component running on the server.

### Installation

**Install Requirements:**

```bash
pip install -r requirements.txt
```

**If you use Conda:**

```bash
conda create -n llm-server python=3.12
conda activate llm-server
pip install -r requirements.txt
```

### Configuration
**Create a `.env` file:**

```.env
# 11434 is ollama default port, change if needed.
GPU_Server_1="http://<Server_IP_1>:11434"
GPU_Server_2="http://<Server_IP_2>:11434"
GPU_Server_3="http://<Server_IP_3>:11434"
```

### Usage
**Run the script:**

```bash
python LLM.py
```

The script will execute, print the JSON responses from the LLM, and save the results to `result.json`.


