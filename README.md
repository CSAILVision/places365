# Pre-release of Places365-CNNs
We release various convolutional neural networks (CNNs) trained on Places365 to the public. Places365 is the latest subset of [Places2 Database](http://places2.csail.mit.edu). There are two versions of Places365: **Places365-Standard** and **Places365-Challenge**. The train set of Places365-Standard has ~1.8 million images from 365 scene categories, where there are at most 5000 images per category. We have trained various baseline CNNs on the Places365-Standard and released them as below. Meanwhile, the train set of Places365-Challenge has extra 6.2 million images along with all the images of Places365-Standard (so totally ~8 million images), where there are at most 40,000 images per category. Places365-Challenge will be used for the Places2 Challenge 2016 to be held in conjunction with the [ILSVRC](http://www.image-net.org/challenges/LSVRC/) and [COCO](http://mscoco.org/dataset/#overview) joint workshop at [ECCV 2016](http://www.eccv2016.org/). 

Places365-Standard and Places365-Challenge will be released at [Places2 website](http://places2.csail.mit.edu) soon.

### Pre-trained CNN models on Places365-Standard:
* AlexNet: ```deploy_alexnet_places365.prototxt``` weights:[http://places2.csail.mit.edu/models_places365/alexnet_places365.caffemodel]
* GoogLeNet: ```deploy_googlenet_places365.prototxt``` weights:[http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel]
* VGG16: ```deploy_vgg16_places365.prototxt``` weights:[http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel]
* VGG16-hybrid1365: ```deploy_vgg16_hybrid.prototxt``` weights:[http://places2.csail.mit.edu/models_places365/vgg16_hybrid.caffemodel]

The category index file is ```categories_places365.txt```. Here we combine the training set of ImageNet 1.2 million data with Places365-Standard to train VGG16-hybrid1365 model, its category index file is ```categories_hybrid1365.txt```. To download all the files, you could access [here](http://places2.csail.mit.edu/models_places365/)

### Performance of the Places365-CNNs
The performance of the baseline CNNs is listed below. We use the class score averaged over 10-crops of each testing image to classify.
<img src="http://places2.csail.mit.edu/models_places365/table2.jpg" alt="Drawing"/>

As comparison, we list the performance of the baseline CNNs trained on [Places205](http://places.csail.mit.edu/downloadCNN.html) as below. There are 160 more scene categories in Places365 than the Places205, the top-5 accuracy doesn't drop much.
<img src="http://places2.csail.mit.edu/models_places365/table1.jpg" alt="Drawing"/>

The performance of the deep features of Places365-CNNs as generic visual features is listed below. The setup for each experiment is the same as the ones in our [NIPS'14 paper](http://places.csail.mit.edu/places_NIPS14.pdf)
![Generic visual feature](http://places2.csail.mit.edu/models_places365/table3.jpg)

Some qualitative prediction results using the VGG16-Places365:
![Prediction](http://places2.csail.mit.edu/models_places365/example_prediction.jpg)

### Reference
Link: [Places2 Database](http://places2.csail.mit,edu), [Places1 Database](http://places.csail.mit.edu)

Please cite the following paper if you use the pre-trained CNN models.
```
Places:10-Million Image Database for Scene Understanding
B. Zhou, A. Khosla, A. Lapedriza, A. Torralba and A. Oliva
Arxiv, 2016 (pdf coming soon)
```
