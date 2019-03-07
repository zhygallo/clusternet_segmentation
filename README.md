# ClusterNet for Unsupervised Segmentation on satellite images
Semantic segmentation on satellite images is used to automatically detect and classify
objects of interest while preserving the geometric structure of these objects. In this work,
we present a novel segmentation approach of weekly supervised learning based on
applying K-Means clustering technique to the high-dimensional output of the Deep
Neural Network. This approach is generic and can be applied to any image and to any
class of objects on that image. At the same time, it does not require the ground-truth
during the training stage.



### Prerequisites

```
Keras 2.2.2
faiss 1.4.0
```

### Installing

All library requirementes needed to run the script are in requirements.txt file

To install them in your local environment use command:

```
pip install -r requirements.txt
```

## Running the experiments

1. You have to generate the train and test (not required) dataset of images of the same size from the original images. For this call tools/patch_generator.py from the root of repository.
For example, to generate images of size 512*512 with no overlap, use the command:
```
python ./tools/patch_generator <input_data_path> --output_folder <output_patches_path> --patch_shape 512 512 --stride 512 512
```

2. To train the keras model in an usupervised fashion run:
```
python ./main.py <input_patches_path>
```
*Note*: in this work we use ImageDataGenerator from keras.preprocessing.image which requires that your images are stored in subfolers one levele deeper that the folder you specified as <input_patches_path>.

## Inspired by
*Deep Clustering for Unsupervised Learning of Visual Features* - [DeepCluster](https://github.com/facebookresearch/deepcluster)



