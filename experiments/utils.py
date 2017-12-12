import torch

def save_model(model_state, filename):
    """ Save model """
    # TODO: add it as checkpoint
    torch.save(model_state,filename)