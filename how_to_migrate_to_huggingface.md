## Deployment Worklist: `openreview_finder` to HF Spaces

**Phase 1: Prepare Your Local Repository**

1.  **✅ Create `app.py`:** Ensure your main Gradio application logic resides in a file named `app.py` at the root of your repository[7].
2.  **✅ Create/Update `requirements.txt`:** Create or verify a `requirements.txt` file listing all Python dependencies[7][6]:
    ```
    gradio
    torch
    transformers
    sentence-transformers # Or the specific library SPECTER2 uses
    chromadb
    openreview-py
    huggingface_hub # Useful for programmatically interacting with HF features if needed
    # Add any other specific dependencies
    ```
3.  **✅ Add `.gitignore`:** Ensure temporary files, caches (`api_cache/`, `chroma_db_persistent/`), virtual environments, etc., are ignored.

**Phase 2: Set Up Hugging Face Space**

4.  **✅ Create Hugging Face Account:** If you don't have one, sign up at Hugging Face[6].
5.  **✅ Create a New Space:**
    *   Go to Hugging Face > Spaces > "Create new Space"[1][6].
    *   Choose an owner (you) and a Space name (e.g., `openreview-finder`).
    *   Select "Gradio" as the Space SDK[1].
    *   Choose visibility (Public recommended for easy access/sharing, Private otherwise)[8][3].
    *   Hardware: Start with "CPU basic - Free". You can upgrade later if needed (e.g., for more RAM)[2][8].
    *   **Crucially: Request Persistent Storage.** Under "Advanced settings" or similar, request at least "Small" persistent storage (15GB) to store the downloaded weights and ChromaDB index permanently[2]. This requires adding a payment method, but free tiers often exist for basic storage.
6.  **✅ Generate Hugging Face Token:**
    *   Go to your Hugging Face Settings > Access Tokens > "New token"[3].
    *   Give it a name (e.g., `github-actions-deploy`).
    *   Assign it the "write" role.
    *   Copy the generated token immediately.

**Phase 3: Configure GitHub Repository & Actions**

7.  **✅ Add HF Token to GitHub Secrets:**
    *   In your `openreview_finder` GitHub repository > Settings > Secrets and variables > Actions[3].
    *   Click "New repository secret".
    *   Name it `HF_TOKEN`.
    *   Paste the Hugging Face token you generated[3].
8.  **✅ Create GitHub Actions Workflow:**
    *   Create the directory `.github/workflows/` in your repository if it doesn't exist.
    *   Create a file named `sync_to_hf_space.yml` (or similar) inside `.github/workflows/`.
    *   Paste the following content, **replacing `` and ``**:

        ```yaml
        name: Sync to Hugging Face Space
        on:
          push:
            branches: [main] # Or your primary branch (e.g., master)

          # Allows you to run this workflow manually from the Actions tab
          workflow_dispatch:

        jobs:
          sync-to-hub:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v3 # Fetches the latest code from your branch
                with:
                  fetch-depth: 0 # Fetch all history for accurate pushes

              - name: Push to HF Space
                env:
                  HF_TOKEN: ${{ secrets.HF_TOKEN }}
                # Replace  and  below
                run: |
                  # Set up Git identity for the push command
                  git config --global user.email "github-actions@github.com"
                  git config --global user.name "GitHub Actions"
                  # Force push the current commit to the main branch of the HF Space repo
                  git push --force https://:$HF_TOKEN@huggingface.co/spaces// main
        ```
        *(Adapted from source[3])*

**Phase 4: Modify Application Code (`app.py`)**

9.  **✅ Adapt `app.py` for Downloads & Persistent Storage:** Modify your script to:
    *   Define paths relative to the execution directory (which should map to persistent storage if configured).
    *   Check if model weights exist in the persistent path.
    *   If weights don't exist, download them from Hugging Face Hub (or their source) and save them to the persistent path.
    *   Initialize ChromaDB using a path within the persistent storage.
    *   Run indexing logic if the index is empty or needs updating.
    *   Launch the Gradio app correctly for the Spaces environment.

    ```python
    import os
    import chromadb
    from sentence_transformers import SentenceTransformer # Or relevant transformer/model loader
    import gradio as gr

    # --- Configuration ---
    # These paths will be relative to the Space's root/persistent storage
    MODEL_DIR = "./persistent_storage/model_weights"
    INDEX_DIR = "./persistent_storage/chroma_db_index"
    MODEL_NAME = "allenai/specter2_base" # Or your specific model identifier on HF Hub

    # --- Ensure Directories Exist ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Ensured directories exist: {MODEL_DIR}, {INDEX_DIR}")

    # --- Model Loading (Download if not present in persistent storage) ---
    model_loaded = False
    if os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")): # Check for a key model file
        try:
            print(f"Attempting to load model from local path: {MODEL_DIR}")
            model = SentenceTransformer(MODEL_DIR)
            model_loaded = True
            print("Model loaded successfully from local storage.")
        except Exception as e:
            print(f"Failed to load model locally ({e}). Will attempt download.")
            # Consider cleaning up MODEL_DIR if loading failed partially

    if not model_loaded:
        try:
            print(f"Downloading model '{MODEL_NAME}'...")
            model = SentenceTransformer(MODEL_NAME)
            print(f"Saving model to {MODEL_DIR}...")
            model.save(MODEL_DIR)
            print(f"Model downloaded and saved successfully.")
        except Exception as e:
            print(f"FATAL: Failed to download or save model '{MODEL_NAME}': {e}")
            # Handle error appropriately - maybe exit or display error in Gradio
            raise e # Re-raise critical error

    # --- ChromaDB Client (Points to persistent storage) ---
    print(f"Initializing ChromaDB client at persistent path: {INDEX_DIR}")
    # Use PersistentClient for saving to disk
    client = chromadb.PersistentClient(path=INDEX_DIR)
    collection = client.get_or_create_collection(name="openreview_papers") # Use your desired name
    print(f"ChromaDB client initialized. Collection '{collection.name}' ready.")

    # --- Indexing Logic Placeholder ---
    # Add your logic here to check if indexing is needed and run it.
    # Example: Index if the collection is empty.
    if collection.count() == 0:
        print("Index is empty. Running indexing process...")
        # placeholder_run_indexing(collection, model) # Replace with your actual function call
    else:
        print(f"Existing index found with {collection.count()} entries.")

    # --- Gradio App Definition ---
    # Replace with your actual Gradio interface definition function
    # def create_interface(db_collection, embedding_model):
    #     # ... your interface components ...
    #     return gr.Interface(...)
    # interface = create_interface(collection, model)

    # Placeholder interface for demonstration
    def greet(name):
        return f"Hello {name}!"
    interface = gr.Interface(fn=greet, inputs="text", outputs="text")
    print("Gradio interface defined.")

    # --- Gradio App Launch ---
    print("Launching Gradio interface...")
    # Use 0.0.0.0 for server_name in containerized environments like Spaces
    interface.launch(server_name="0.0.0.0", server_port=7860)
    print("Gradio App Launched and running.")

    ```
10. **✅ Add Secret Handling (If Needed):** If using OpenReview credentials or other secrets:
    *   Add them as Secrets in the Hugging Face Space settings[3][8].
    *   Access them in `app.py` using `os.environ.get("YOUR_SECRET_NAME")`.

**Phase 5: Deploy and Test**

11. **✅ Commit and Push Changes:** Commit all new/modified files (`app.py`, `requirements.txt`, `.github/workflows/`, `.gitignore`) to your GitHub repository's `main` branch[6].
    ```bash
    git add .
    git commit -m "Configure for Hugging Face Spaces deployment with runtime downloads"
    git push origin main
    ```
12. **✅ Monitor GitHub Action:** Check the "Actions" tab in your GitHub repository for the "Sync to Hugging Face Space" workflow. Verify it completes successfully[3].
13. **✅ Monitor Space Build & Logs:** Go to your Hugging Face Space page. Watch the "Build" and "Runtime" logs. Ensure dependencies install (`requirements.txt`), the model downloads (first time), ChromaDB initializes, and the Gradio app starts without errors[8][6].
14. **✅ Test Application:** Access the Space's URL. Test functionality thoroughly. Check logs for any runtime issues.
15. **✅ Iterate:** Debug any issues by examining Space logs, adjusting code, pushing changes to GitHub, and monitoring the automatic redeployment. Consider upgrading hardware if you hit resource limits[8].

Citations:
[1] https://huggingface.co/docs/hub/en/spaces-overview
[2] https://huggingface.co/docs/hub/en/spaces
[3] https://docs.evidence.dev/deployment/self-host/hugging-face-spaces
[4] https://huggingface.co/docs/hub/en/spaces-sdks-docker-evidence
[5] https://evidence.dev/blog/hugging-face-spaces
[6] https://www.marqo.ai/blog/how-to-create-a-hugging-face-space
[7] https://www.youtube.com/watch?v=NM37r6v3sFw
[8] https://docs.zenml.io/getting-started/deploying-zenml/deploy-using-huggingface-spaces
