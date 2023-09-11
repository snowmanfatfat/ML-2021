import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))