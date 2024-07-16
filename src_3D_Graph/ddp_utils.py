import torch
import torch.distributed as dist

import os


# Initialization for distributed training
def init_process_group():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    dist_url = 'env://'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        rank, dist_url, gpu), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
    
def gather_lists(local_list):
    rank = get_rank()
    world_size = get_world_size()
    # Convert local list to tensor
    local_tensor = torch.tensor(local_list, dtype=torch.float32).to(rank)
    
    # Gather sizes of tensors from all processes
    local_size = torch.tensor([len(local_tensor)], dtype=torch.int64).to(rank)
    sizes = [torch.tensor([0], dtype=torch.int64).to(rank) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    
    # Calculate maximum size for padding
    max_size = max(size.item() for size in sizes)
    
    # Pad local tensor to maximum size
    padded_tensor = torch.zeros(max_size, dtype=torch.float32).to(rank)
    padded_tensor[:local_size] = local_tensor
    
    # Gather all tensors from all processes
    gathered_tensors = [torch.zeros(max_size, dtype=torch.float32).to(rank) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_tensor)
    
    # Unpad and combine the gathered tensors
    combined_list = []
    for size, tensor in zip(sizes, gathered_tensors):
        combined_list.extend(tensor[:size].tolist())
    
    return combined_list


    
    