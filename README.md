## Bee Classification using Convolutional Neural Network 
###### A system for classifying bees (honey bee or a bumble bee).

#### Summary
This is done to compete in [`The Metis Challenge: Naive Bees Classifier`](http://www.drivendata.org/competitions/8/) competition.

- The process_data reads the images, scales it, creates synthetic dataset to make the classifier work well by creating reflected and translated versions of images. It then takes care of class imbalance by undersampling this enhanced dataset. All this processes data is then stored in numpy arrays.
- The CNN architecture is built/run on Theano making use of abstraction libraries like Lasagne.
- I've experimented with various architectures (will upload the configurations soon!)

#### Code
The following modules are required to run the system:

  * Python 2.7
  * NumPy
  * Theano
  * Lasagne
  * Pandas
  * scikit-learn
  * Matplotlib