Test Workflow


In the repo you can find multiple files to get the energy score for in and out distribution for CIFAR-10(in-distribution) and SVHN(out-distribution). In the wrn.py file a WideResNet model is implemented as a network to capture the relevant features from the CIFAR-10 and SVHN dataset. Then the dataloader.py contains only loading both the datasets and preprocessing them for the network. In train.py and test.py we train the model on the CIFAR-10 dataset which gives around 91% accuracy and is tested on SVHN dataset which gives 10% accuracy(expected). In the main.py file the energy score is calculated and the graph is plotted

The energy score for the data is calculated from the input of the features as logits to the energy function and prints the energy score based on the number of samples
Training and validation loss are also calculated to make sure the results are correct.


In the Python code waymo_features, To run the code, first clone the waymo open dataset GitHub repo and change the path to the frame dataset
https://github.com/waymo-research/waymo-open-dataset
There are 2 functions show_camera_image() and show_range_image() which display the camera images with their bounding boxes, and range images 
which display the range images of range, intensity, and elongation. Also, print the frame details. These can be used as features for the energy function as logits to print and in and out
distribution. 

