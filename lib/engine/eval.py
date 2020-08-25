import torch
from torch.nn import DataParallel
from tqdm import tqdm

def evaluate(model, device, dataloader, evaluator):
    """
    Test engine.
    
    :param model: model(data) -> results
    :param dataloader: next(iter(dataloader)) -> data, targets
    """
    
    # cpu = torch.device('cpu')
    model = model.to(device)
    if isinstance(model, DataParallel):
        model = model.module
        
        
    # Very important!
    evaluator.reset()
    # with torch.no_grad() can not be used here
    model.eval()
    it = 0
    pbar = tqdm(dataloader, total=100)
    for data in pbar:
        if it > 100: break
        it += 1
        data = list(data)
        # image to device
        data[0] = data[0].to(device)
        evaluator.evaluate(model, data)
        pbar.set_description(evaluator.get_results())
            
    print('Final: ', evaluator.get_results())
        
    # return evaluator.get_results()
    
    
