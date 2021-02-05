
def get_prediction_function(model):

    #pytorch
    if 'nn.Module' in str(type(model)):

        import torch
        from torch.autograd import Variable
        import torch.nn.functional as F

        def _predict_with_pytorch(model, data):
            model.eval()

            with torch.no_grad():
                patch = Variable(torch.Tensor(data).float())
                patch = patch.to(model.device())
                patch = F.softmax(model(patch), dim=1).cpu().numpy()
            return patch

        return _predict_with_pytorch

    #keras
    elif 'keras' in str(type(model)).lower():


        def _predict_with_keras(model, data):
            return model.predict(data)

        return _predict_with_keras

    else:
        raise NotImplementedError()



