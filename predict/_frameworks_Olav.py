
def get_prediction_function(model):

    import torch
    from torch.autograd import Variable
    import torch.nn.functional as F

    def _predict_with_pytorch(model, data, device):
        model.eval()

        with torch.no_grad():
            patch = torch.Tensor(data).float()
            patch = patch.to(device)
            patch = F.softmax(model(patch), dim=1).cpu().numpy()
        return patch

    return _predict_with_pytorch