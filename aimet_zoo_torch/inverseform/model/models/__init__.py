# pylint: skip-file
"""
Network Initializations
"""

import importlib
import torch

# from aimet_zoo_torch.inverseform.model.utils.config import cfg


# def get_net(arch, criterion):
#     """
#     Get Network Architecture based on arguments provided
#     """
#     net = get_model(network='models.' + arch,
#                     num_classes=cfg.DATASET.NUM_CLASSES,
#                     criterion=criterion, has_edge=cfg.LOSS.edge_loss)
#     num_params = sum([param.nelement() for param in net.parameters()])

#     net = net.cuda()
#     return net


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex

        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion, has_edge=False):
    """
    Fetch Network Function Pointer
    """
    module = network[: network.rfind(".")]
    model = network[network.rfind(".") + 1 :]
    mod = importlib.import_module(module)

    net_func = getattr(mod, model)
    if not has_edge:
        net = net_func(num_classes=num_classes, criterion=criterion)
    else:
        net = net_func(
            num_classes=num_classes, criterion=criterion, has_edge_head=has_edge
        )
    return net
