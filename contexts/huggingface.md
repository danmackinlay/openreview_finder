Based on the provided documentation, here is a concise summary of the steps and configurations needed to deploy your `openreview_finder` repository from GitHub to Hugging Face Spaces:

## Hugging Face Spaces Deployment Guide Summary

**1. Create the Space:**
*   Go to Hugging Face, click "Spaces," then "Create New Space"[4].
*   Alternatively, use the `huggingface_hub` Python library:
    ```python
    from huggingface_hub import HfApi
    api = HfApi()
    # Replace 'your-username' and 'your-space-name'
    repo_id = "your-username/your-space-name"
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
    ```

*   Specify "Gradio" as the SDK during creation[2][6].

**2. Structure Your Repository:**
*   **`app.py`:** This file is essential. It should contain your Gradio application code, initializing and launching the interface[4].
*   **`requirements.txt`:** List all Python dependencies required by your application (e.g., `gradio`, `torch`, `transformers`, `chromadb`, `openreview-py`)[4]. Hugging Face Spaces uses this to install packages.
*   **Other Files:** Include your script modules, data files, and any other necessary assets.

**3. Configure the Space:**
*   **Hardware:** Select appropriate hardware (CPU Basic, Premium, GPU). You can manage this via the Space settings UI or programmatically[2][6]. Consider upgrading if the free tier memory is insufficient.
*   **Storage:** For persistent storage (needed for your index and weights), request a storage tier (e.g., Small, Large). This can be done during creation or later via the UI or API[2]. Note that you cannot downgrade storage without deleting it first[2].
    ```python
    from huggingface_hub import SpaceStorage
    # Request storage after creation
    api.request_space_storage(repo_id=repo_id, storage=SpaceStorage.LARGE)
    # Or request storage during creation
    api.create_repo(..., space_storage="large")
    ```

*   **Secrets:** If your application needs API keys or other sensitive information (like an OpenReview password if not using anonymous access), add them as secrets in the Space settings UI or via the `huggingface_hub` library[2][6]. Access them as environment variables within your `app.py`.

**4. Push Your Code:**
*   **Method 1: Git Clone & Push (Manual)**
    1.  Install Git if you haven't already[4].
    2.  Clone the empty Space repository: `git clone https://huggingface.co/spaces/your-username/your-space-name`[4].
    3.  Copy your project files (`app.py`, `requirements.txt`, etc.) into the cloned directory[4].
    4.  Commit and push the changes:
        ```bash
        git add .
        git commit -m "Initial application commit"
        git push
        ```

*   **Method 2: Syncing from GitHub (Manual or Automated)**
    1.  Create a Hugging Face access token with write permissions in your Hugging Face account settings[5].
    2.  Add the Hugging Face space as a remote to your local GitHub repository:
        ```bash
        git remote add space https://huggingface.co/spaces//
        # Use token for authentication (replace  and user/repo names)
        git remote set-url space https://:@huggingface.co/spaces//
        ```

    3.  Push your desired branch (e.g., `main`) to the Space remote: `git push --force space main`[5].
    4.  **Automation with GitHub Actions:**
        *   Add your Hugging Face token as a secret named `HF_TOKEN` in your GitHub repository's settings (Settings > Secrets and variables > Actions)[1][5].
        *   Create a workflow file (e.g., `.github/workflows/sync_to_hf.yml`) in your GitHub repository[1][5]:
            ```yaml
            name: Sync to Hugging Face hub
            on:
              push:
                branches: [main] # Or your default branch
              workflow_dispatch: # Allows manual triggering
            jobs:
              sync-to-hub:
                runs-on: ubuntu-latest
                steps:
                  - uses: actions/checkout@v3
                    with:
                      fetch-depth: 0
                      lfs: true # Important if you have large files managed by Git LFS
                  - name: Push to hub
                    env:
                      HF_TOKEN: ${{ secrets.HF_TOKEN }}
                      # Replace  and
                    run: git push --force https://:$HF_TOKEN@huggingface.co/spaces// main
            ```
 (Adapted slightly from source 5 for clarity)

**5. Tools:**
*   **`huggingface-cli`:** Command-line tool for tasks like logging in and uploading files[1].
*   **`huggingface_hub`:** Python library for programmatically interacting with the Hub (creating repos, managing settings)[2][3][6].

After pushing your code and configuring the space, Hugging Face will build the environment based on `requirements.txt` and run your `app.py` file[4].

Citations:
[1] https://docs.evidence.dev/deployment/self-host/hugging-face-spaces
[2] https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces
[3] https://huggingface.co/docs/huggingface_hub/main/en/guides/repository
[4] https://www.marqo.ai/blog/how-to-create-a-hugging-face-space
[5] https://gist.github.com/Hansimov/6002fddd5f7a49c210ed1b3757acb271
[6] https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-spaces
[7] https://huggingface.co/docs/hub/en/spaces-overview
[8] https://huggingface.co/docs/hub/en/spaces
[9] https://huggingface.co/docs/chat-ui/en/installation/spaces
[10] https://drlee.io/huggingface-spaces-a-beginners-guide-to-creating-your-first-space-for-data-science-935d79a4a37b
[11] https://www.youtube.com/watch?v=OqqBk8_4gmQ
[12] https://docs.v1.argilla.io/en/v1.28.0/getting_started/installation/deployments/huggingface-spaces.html
[13] https://github.com/ruslanmv/How-to-Sync-Hugging-Face-Spaces-with-a-GitHub-Repository
[14] https://www.geeksforgeeks.org/huggingface-spaces-a-beginners-guide/
[15] https://huggingface.co/docs/hub/en/storage-backends
[16] https://huggingface.co/docs/hub/en/repositories-getting-started
[17] https://docs.zenml.io/getting-started/deploying-zenml/deploy-using-huggingface-spaces
[18] https://huggingface.co/docs/hub/en/spaces-config-reference
[19] https://docs.langflow.org/deployment-hugging-face-spaces
[20] https://docs.v1.argilla.io/en/v1.19.0/getting_started/installation/deployments/huggingface-spaces.html
