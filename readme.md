## LlamaIndex RAG
- Embedding model is running in a RunPod pod
- LLM model is running in another RunPod pod
- ChromaDB is installed locally

### Model Serving from RunPod Steps
1. Create a Runpod pod
2. Open up 8000 port in the pod configuration
3. Configure vLLM in the pod

    -  UV Install - `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - vLLM Install
        - `uv venv --python 3.12 --seed`
        - `source .venv/bin/activate`
        - `uv pip install vllm --torch-backend=auto`

4. Serve Embedding Model `BAAI/bge-small-en-v1.5` using `nohup vllm server BAAI/bge-small-en-v1.5`
5. Serve Qwen Model `Qwen/Qwen2.5-Coder-7B-Instruct` using `nohup vllm server Qwen/Qwen2.5-Coder-7B-Instruct`

Follow the [/Users/potter/Documents/MLArena/MLOps-Platform-Project/Hands-On/hfagnt-2-2-the-llamaindex-framework/components-of-llamaindex.ipynb](https://github.com/mlops-pltf/mlaist-54/tree/feature/hfagnt-2-2-the-llamaindex-framework) to see how to use runpod served models in llama pipeline

### To Do
- Use serverless ChromaDB
- Automate the Model Inference Nodes