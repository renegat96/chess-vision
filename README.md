# Stanford CS230 final project

Adam Stanford-Moore and Hristo Stoyanov built this for the Spring 2019 edition
of the CS230 class.

Files in this respository:

best_model.h5 :  
These are the saved weights of the Keras model trained with batch-size 256, 20 epochs, on 100 unfrozen layers of the ResNet50 (with modified last layer to be 13 classes).

crop.py  :  
This is the main file that takes in a photo of a board and outputs the FEN notation using best_model.h5
Use:
python3 crop.py board_image.jpg

chess_vision.py  : 
This script loads the data, builds adn trains a model and makes several plots for analysis. must specify an OUTFOLDER inside for the location of the plots 

data   :   
Labelled chess piece pieces we collected and trained our model on in addition to data from Daylen Yang(https://github.com/daylen/chess-id)
