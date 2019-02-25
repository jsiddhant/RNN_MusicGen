import torch

def setup_device():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        computing_device = torch.device('cuda')
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    return computing_device
