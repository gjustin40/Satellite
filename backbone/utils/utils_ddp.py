import torch
import torch.distributed as dist

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def average_gradients(model, world_size):
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def reduce_dict(input_dict, world_size, average=True, cpu=False):   
    names, values = [], []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
        
    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size
        
    reduced_dict = {}
    for k, v in zip(names, values):
        if cpu:
            v = v.cpu()
        reduced_dict[k] = v

    return reduced_dict

def cleanup(rank):
    # dist.cleanup()  
    dist.destroy_process_group()
