# Vehicle-Classification

This project aims to develop a real-time vehicle classification system using AI and NVIDIA Jetson. The system can distinguish between different types of vehicles (e.g., cars, trucks, motorcycles) based on images from cameras. This project enhances traffic management, toll collection, and smart city applications by providing accurate and efficient vehicle classification.
The project involves training a CNN model on a labeled dataset of vehicle images. Once trained, the model is deployed on an NVIDIA Jetson device, which uses its GPU capabilities to perform real-time inference. The system processes video feeds from cameras, classifies the vehicles in the footage, and provides instant feedback, enhancing the efficiency and effectiveness of traffic monitoring systems.

![Screenshot 2024-08-07 113347](https://github.com/user-attachments/assets/40b38a9c-5407-454c-93c4-97a630597c5d)


## The Algorithm

The core of the project is a Convolutional Neural Network (CNN) designed for image classification. Here's a detailed explanation of how the algorithm works:

Convolutional Layers: These layers apply various filters to the input images to detect features such as edges, textures, and shapes. Each filter slides over the image, creating feature maps that represent different aspects of the input.

Pooling Layers: After the convolutional layers, pooling layers (like max pooling) reduce the spatial dimensions of the feature maps. This process makes the computation more efficient and the model more robust to small variations in the input.

Fully Connected Layers: The feature maps are then flattened and passed through fully connected layers. These layers combine the extracted features to make the final classification decisions.

Activation Functions: Non-linear functions, such as ReLU (Rectified Linear Unit), introduce non-linearity into the model, allowing it to learn complex patterns.

Training Process: The network is trained on a labeled dataset of vehicle images. The training process involves adjusting the model's weights using backpropagation to minimize the loss function, which measures the difference between the predicted and actual labels.

Deployment: The trained model is deployed on an NVIDIA Jetson device. The Jetson's GPU accelerates the inference process, enabling real-time vehicle classification from camera feeds.

## Running this project


Set up an SSH conection with your Jetson Nano.

Open a new terminal.

Run this command to update your installer: sudo apt-get update

Enter your password to continue.

Run this command to install cmake: sudo apt-get install git cmake

Clone the jetson-inference project: git clone --recursive https://github.com/dusty-nv/jetson-inference

Change into the newly created jetson-inference folder: cd jetson-inference

Update the contents of the folder: git submodule update --init

Install the python processes necessary to run the AI: sudo apt-get install libpython3-dev python3-numpy

Switch to the jetson-inference directory and make a build directory to build your project into: mkdir build

Switch to the build directory: cd build

Build the project with this command: cmake ../

Switch to the build directory if not already in it. Run this command to run make the python files: make

Run this command to install make. sudo make install

Configure the make command: sudo ldconfig.

Download the skin disease dataset at https://www.kaggle.com/datasets/marquis03/vehicle-classification/data

Extract the elements of the zip file and drag the extracted folder into jetson-inference > python > training > classification > data

cd back into jetson-inference.

Run this command to allot more memory for the program: echo 1 | sudo tee /proc/sys/vm/overcommit_memory

Run the docker with this command: ./docker/run.sh

Enter your password when prompted.

cd into jetson-inference/python/training/classification.

Transfer the files from the computer to Visual Studio Code in data folder (drag and drop)

Run this code to start training the AI based on the dataset: python3 train.py --model-dir=models/VehicleClassification data/vehicles --epochs=35

To increase or decrease the amount of training the AI model receives, increase or decrease the number in --epochs=35 respectively.

Once the model has finished training, cd into jetson-inference/python/training/classification.

Export the model into onnx: python3 onnx_export.py --model-dir=models/vehicle classification-dataset

Use ctrl+D to exit the docker.

cd into jetson-inference/python/training/classification.

Run these commands to set up variables needed for image processing:

NET=models/Fproject
DATASET=data/Fproject
Run this command try an image from the test folder. Change 'NAME HERE' to name your output file, rename 'NAME OF CATEGORY' to the category of what you want to test, rename 'IMAGE NAME' to the name of the image. You can rename the image first in the side menu to customize the name. imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/NAME OF CATEGORY/IMAGE NAME .jpg $DATASET/output/OUTPUT NAME.jpg
Here is an example of what your command should look like: imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/Fproject/Firetruck1.jpg Firetest1.jpg

The results should automatically go into the classification folder, and double click a picture to view the AI's classification and confidence.






[View a video explanation here](video link)
