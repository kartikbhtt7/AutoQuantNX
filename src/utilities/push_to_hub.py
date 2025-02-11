import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def push_to_hub(local_path: str, repo_name: str, hf_token: str, commit_message: Optional[str] = None, tags: Optional[list] = None) -> Tuple[Optional[str], str]:
    """
    Pushes a folder containing model files to the HuggingFace Hub.

    Args:
        local_path (str): Local directory containing the model files to upload.
        repo_name (str): The repository name (not the full username/repo_name).
        hf_token (str): HuggingFace authentication token.
        commit_message (str, optional): Commit message for the upload.
        tags (list, optional): Tags to include in the model card.

    Returns:
        Tuple[Optional[str], str]: (repository_name, status_message)
    """
    try:
        api = HfApi(token=hf_token)

        # Validate token
        try:
            user_info = api.whoami()
            username = user_info["name"]
        except Exception as e:
            return None, f"❌ Authentication failed: Invalid token or network error ({str(e)})"

        # Full repository name with the username
        full_repo_name = f"{username}/{repo_name}"
        
        # Create the repo
        try:
            create_repo(full_repo_name, token=hf_token, exist_ok=True)
            logger.info(f"Repository created/verified: {full_repo_name}")
        except Exception as e:
            return None, f"❌ Repository creation failed: {str(e)}"

        # Create model card
        try:
            tags_list = tags or []
            tags_section = "\n".join(f"- {tag}" for tag in tags_list)
            model_card = f"""---
tags:
{tags_section}
library_name: optimum
---

# Model - {repo_name}

This model has been optimized and uploaded to the HuggingFace Hub.

## Model Details
- Optimization Tags: {', '.join(tags_list)}
"""
            with open(os.path.join(local_path, "README.md"), "w") as f:
                f.write(model_card)
        except Exception as e:
            logger.warning(f"Model card creation warning: {str(e)}")

        # Upload the folder
        try:
            api.upload_folder(
                folder_path=local_path,
                repo_id=full_repo_name,
                repo_type="model",
                commit_message=commit_message or "Upload optimized model"
            )
            success_msg = f"✅ Model successfully pushed to: {full_repo_name}"
            logger.info(success_msg)
            return full_repo_name, success_msg
        except Exception as e:
            error_msg = f"❌ Upload failed: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    except Exception as e:
        error_msg = f"❌ Unexpected error during push: {str(e)}"
        logger.error(error_msg)
        return None, error_msg