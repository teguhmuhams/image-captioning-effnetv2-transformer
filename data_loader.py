from torch.utils.data import DataLoader

def get_loader(train_dataset = None, val_dataset = None, test_dataset = None, batch_size = None, num_workers = None):
    
    train_loader = val_loader = test_loader = None
    
    if train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader