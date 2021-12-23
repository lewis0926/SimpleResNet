import matplotlib.pyplot as plt
import torch


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(entity, device):
    if isinstance(entity, (list, tuple)):
        return [to_device(element, device) for element in entity]
    return entity.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for b in self.dataloader:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dataloader)
