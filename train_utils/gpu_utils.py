# 🖥️ Log GPU status
def get_device_summary():
    import subprocess

    print("=== NVIDIA-SMI ===")
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Failed to run nvidia-smi: {e}")

    import torch

    print("[INFO] PyTorch version:      ", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print("[INFO] CUDA available:       ", torch.cuda.is_available())
    print("[INFO] CUDA toolkit version: ", torch.version.cuda)
    print("[INFO] Device count:         ", torch.cuda.device_count())
    if torch.cuda.is_available():
        print(f"[INFO] Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - CUDA:{i} — {torch.cuda.get_device_name(i)}")
    else:
        print("[INFO] Only CPU is available.")
    return device
