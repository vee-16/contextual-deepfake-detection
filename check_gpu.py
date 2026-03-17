"""Quick check that PyTorch sees and can use the GPU."""
import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device count:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))
        # Run a tiny op on GPU to confirm it works
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x
        print("GPU compute test: OK")
    else:
        print("Running on CPU only.")

if __name__ == "__main__":
    main()
