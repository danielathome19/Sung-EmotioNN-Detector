# About
EmotioNN is a convolutional neural network (CNN) trained to classify emotions in singing voices.
To find out more, check out the provided research paper:
  * "Emotion Recognition of the Singing Voice: Toward a Real-Time Analysis Tool for Singers" (DOI:10.48550/arxiv.2105.00173) 
  * Also contained in the "PaperAndPresentation" folder is a handout note, the research paper, and presentation of the research.

# Usage
See:
 * https://www.youtube.com/watch?v=f9hs8TYyBxU for an overview of analyzing the output data.
* https://www.youtube.com/watch?v=dsruK0GctG4 for a demonstration of the program and features.


**NOTE:** these folders should be placed in the **same** folder as "main.py". For folder existing conflicts, simply merge the directories.

In main.py, the "fmain" function acts as the controller for the model, where calls to train the model, create a prediction, split a wave file, isolate vocals, test in realtime, and all other functions are called. One may also call these functions from an external script ("from main import wavsplit", etc.).

To choose an operation or series of operations for the model to perform, simply edit the main function before running. Examples of all function calls can be seen commented out within main.

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.
