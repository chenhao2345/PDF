import torch
import torch.distributed as dist


# class GatherLayer(torch.autograd.Function):
#     """Gather tensors from all process, supporting backward propagation."""
#
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
#         dist.all_gather(output, input)
#
#         return tuple(output)
#
#     @staticmethod
#     def backward(ctx, *grads):
#         (input,) = ctx.saved_tensors
#         grad_out = torch.zeros_like(input)
#
#         # dist.reduce_scatter(grad_out, list(grads))
#         # grad_out.div_(dist.get_world_size())
#
#         grad_out[:] = grads[dist.get_rank()]
#
#         return grad_out


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)