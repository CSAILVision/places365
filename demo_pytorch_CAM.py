import torch
from torch.autograd import Variable as V
import torchvision.models as models
import skimage.io
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
# function to load exif of image
from PIL import Image, ExifTags

def imreadRotate(fn):
    image=Image.open(fn)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        print('dont rotate')
        pass
    return image

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Scale((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'whole_wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu
    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()


# retrieve and predict the uploaded images
root_path = '/data/vision/torralba/scratch2/bzhou/places365demo_upload'
sourceFolder = root_path + '/source';
resultFolder = root_path + '/result';
moveFolder = root_path + '/processed';
segmentationFolder = root_path + '/segmentation';

import glob
import time
# first clean up the uploaded images (from last crash)
images = glob.glob(sourceFolder + '/*.jpg')
for imgfile in images:
    os.remove(imgfile)
    print('delete ' + imgfile)

print('standby ...')
num_total = 0
time_start = time.strftime('%Y-%m-%d %H:%M')
while 1:
    time.sleep(1)
    images = glob.glob(sourceFolder + '/*.jpg')
    for imgfile in images:
        try:
            del features_blobs[:]
            print('processing ' + imgfile)
            file_id = imgfile.split('/')[-1][:-4]
            file_json_tmp = '%s/%s_tmp.json' % (resultFolder, file_id)
            file_json ='%s/%s.json' % (resultFolder, file_id)
            if os.path.isfile(file_json):
                print('prediction exist ' + file_json)
                os.remove(imgfile)
                pass
            num_total = num_total + 1
            file_segmentation = '%s/%s.jpg' % (segmentationFolder, file_id)
            # check mask file, if exists then delete
            if os.path.isfile(file_segmentation):
                os.remove(file_segmentation)

            img = imreadRotate(imgfile)
            input_img = V(tf(img).unsqueeze(0), volatile=True)

            # forward pass
            logit = model.forward(input_img)
            h_x = F.softmax(logit).data.squeeze()
            probs, idx = h_x.sort(0, True)

            #output json file
            fid = open(file_json_tmp, 'w')
            fid.write('{')
            # output the IO prediction
            io_image = np.mean(labels_IO[idx[:10].numpy()]) # vote for the indoor or outdoor
            if io_image < 0.5:
                print('--TYPE OF ENVIRONMENT: indoor')
                fid.write('"type": "indoor", ')
            else:
                print('--TYPE OF ENVIRONMENT: outdoor')
                fid.write('"type": "outdoor", ')

            # output the prediction of scene category
            out = []
            for i in range(0, 5):
                print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                if i==0 or probs[i]>0.10:
                    out.append('%s (%.3f)' % (classes[idx[i]], probs[i]))
            fid.write('"scenes": "%s", ' % (', '.join(out)))
            fid.write('"topcategory": "%s", ' % (classes[idx[0]]))

            # output the scene attributes
            responses_attribute = W_attribute.dot(features_blobs[1])
            idx_a = np.argsort(responses_attribute)
            print('--SCENE ATTRIBUTES:')
            out = ', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)])
            print(out)
            fid.write('"attributes": "%s", ' % out)
            fid.write('"segmentation": "%s.jpg"' % file_id)

            fid.write('}')
            fid.close()
            os.rename(file_json_tmp, file_json)
            print('json file saved to ' + file_json)
            # generate class activation mapping
            print('CAM as ' + file_segmentation)
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

            # render the CAM and output
            img = cv2.imread(imgfile)
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.4 + img * 0.5
            result = cv2.resize(result, (int(width*300/height), 300))
            cv2.imwrite(file_segmentation, result)
            time_now = time.strftime('%Y-%m-%d %H:%M')
            print('from %s to %s: processed image number: %d' % (time_start, time_now, num_total))
#            os.remove(imgfile)
        except:
            print('shit happens')
            if os.path.isfile(imgfile):
                os.remove(imgfile)

