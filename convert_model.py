import torch
from torch.autograd import Variable as V
import torchvision.models as models
from PIL import Image
from torchvision import transforms as trn
from torch.nn import functional as F
import os

# th architecture to use
arch = 'resnet18'
model_weight = '/data/vision/torralba/deepscene/moments/models/2stream-simple/model/kinetics_rgb_resnet18_2d_single_stack1_fromscratch_best.pth.tar'
model_name = 'resnet18_kinetics_fromscratch'

# create the network architecture
model = models.__dict__[arch](num_classes=400)

#model_weight = '%s_places365.pth.tar' % arch

checkpoint = torch.load(model_weight, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} # the data parallel layer will add 'module' before each layer name
model.load_state_dict(state_dict)
model.eval()

model.cpu()
torch.save(model, 'whole_' + model_name + '.pth.tar')
print('save to ' + 'whole_' + model_name + '.pth.tar')
