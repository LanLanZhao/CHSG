import torch.optim as optim


def load_scheduler(model_name, model, args):
    if model_name != "CHSG":
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[150, 225],
        gamma=0.1,
    )
    return optimizer, scheduler
