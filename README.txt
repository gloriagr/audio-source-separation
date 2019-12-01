Structure of the code:

1. pre_processing.py prepares the data according to the article:
    It performs STFT on short time frames of all the tracks, and saves their magnitude and phase locally.

2. pre_processing_2.py prepares the data for experiments 2-8:
    It simply reorders the files so there will be 80 training songs instead of 50
    
3. data_loader.py loads the magnitudes that were created in the previous scripts on demand (generator).
    It uses caching for speeding.

4. build_model_original.py is the network that was presented in the original article.
    build_model_dropout.py is the network above, only we've added a dropout layer to the first fully connected layer.
    build_model_maxpool.py is the network of experiment 5: 
    We added a max pooling layer after the first convolutional layer, and added max unpooling in the decoding part.

5. train_model.py implements the training and validation of the models.
    After each epoch, it saves the current weights of the model in the "Weights" folder.

6. test_model.py loads a weights file and runs this model on the test set.
    It forwards the results to post_processing.py

7. post_processing.py multiplies these results by the phase, and then performs inverse STFT to get an audio file.
    It saves the results locally.

8. stitching.py stitches the short audio results into full tracks.

9. evaluate.py reorders the results and then evaluates SDR, SIR, SAR by comparing the results and the original tracks.
    It calculates the mean and standard deviation over all the test songs.

***************************************************************

Requirements:

* librosa
* numpy
* pytorch
* os
* re
* shutil
* natsort
* glob
* mir_eval
* scipy.io

***************************************************************
Preparing the Data:

1. Download the full DSD100 dataset from the link below:
https://sigsep.github.io/datasets/dsd100.html

2. Unzip it to this folder.

3. Go to the sub-folder "src" and run the scripts (in this order):
pre_processing.py
pre_processing_2.py

***************************************************************
Instructions:

1. Choose the appropriate group of directories in the following scripts, for the needed experiment:
- data_loader.py
- test_model.py
- stitching.py
- evaluate.py

2. In evaluate.py, also make sure you choose the right lists l1, l2.

3. In train_model.py, make sure you are importing the correct model script (original, maxpool or dropout).

4. Run train_model.py.

5. In the Weights folder that was created, copy the name of the weights file you're interested in, 
    and paste it in load_state_dict in test_model.py.

6. Run test_model.py.

7. Run stitching.py.

8. Run evaluate.py.
