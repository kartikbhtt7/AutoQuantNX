import os
import shutil
import psutil
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceManager:
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dirs = {
            "onnx": self.temp_dir / "onnx_output",
            "quantized": self.temp_dir / "quantized_models",
            "cache": self.temp_dir / "model_cache"
        }
        self.setup_directories()

    def setup_directories(self):
        for dir_path in self.temp_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def cleanup_temp_files(self, specific_dir: Optional[str] = None) -> str:
        try:
            if specific_dir:
                if specific_dir in self.temp_dirs:
                    shutil.rmtree(self.temp_dirs[specific_dir], ignore_errors=True)
                    self.temp_dirs[specific_dir].mkdir(exist_ok=True)
            else:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.setup_directories()
            return "✨ Cleanup successful!"
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return f"❌ Cleanup failed: {str(e)}"

    def get_memory_info(self) -> Dict[str, float]:
        vm = psutil.virtual_memory()
        memory_info = {
            "total_ram": vm.total / (1024 ** 3),
            "available_ram": vm.available / (1024 ** 3),
            "used_ram": vm.used / (1024 ** 3)
        }
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_info.update({
                "gpu_total": torch.cuda.get_device_properties(device).total_memory / (1024 ** 3),
                "gpu_used": torch.cuda.memory_allocated(device) / (1024 ** 3)
            })
        
        return memory_info
