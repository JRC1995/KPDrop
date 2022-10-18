import torch as T


def get_scheduler(config, optimizer):
    if config["scheduler"] is None:
        return None, False
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=config["scheduler_reduce_factor"],
                                                           patience=config["scheduler_patience"])
        return scheduler, True
