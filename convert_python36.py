import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F


archs = ['resnet50','densenet161','alexnet']
for arch in archs:
    model_file = 'whole_%s_places365.pth.tar' % arch
    save_file = 'whole_%s_places365_python36.pth.tar' % arch

    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    torch.save(model, save_file)
    print('converting %s -> %s'%(model_file, save_file))

