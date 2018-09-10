# LicensePlateDetector
Detects license plate of car and recognizes its characters

## Method

1. Detect License Plate
2. Perform segmentation of characters
3. Train a ML model to predict characters
4. Prediction of characters in License Plate

The approach used to segment the images is Connected Component Analysis. Connected regions wil imply that all the connected pixels belong
to the same object. A pixel is said to be connected to another if they both have the same value and are adjacent to each other.

Car Image -> Grayscale Image -> Binary Image -> Applying CCA to get connected regions -> Detect license plate out of all connected regions
(Assumptions made : width of the license plate region to the full image ranges between 15% and 40% and height of the license plate region
to the full image is between 8% & 20%)

Output of first step is a license plate image detected in a car image. This is provided as input to step2 and CCA is applied on this image
to bound the characters in plate.Each character identified is appended into a list.

Model is trained using SVC (4 cross fold validation) with dataset present in directory train20X20. The model is saved as finalized_model.sav
which is then loaded to predict each character.

Once the characters of plate is obtained and model is trained, the model is loaded in order to predict each character.


## Setup

Clone the repository.
Change the path of the image/video file in DetectPlate.py 
Run PredictCharacters.py. This will load the trained model (finalized_model.sav) which is added to repo for reference. Your own model can 
also be trained using the dataset attached in repo.
Running PredictCharacters.py first gives grayscale and binary image. Then produces gray image with license plate bounded inside a rectangle.
Each characters are also segmented and shown within boxes.
Finally the model predicts the license plate.


## Screenshots

1. Car Video 

>Original frame of video


![fram35](https://user-images.githubusercontent.com/19779081/45311494-e3dc3d80-b546-11e8-86b3-ea1815f8e7f8.jpg)

>License Plate Detected

![out](https://user-images.githubusercontent.com/19779081/45310686-9e1e7580-b544-11e8-8445-5d18496e23ba.png)

>Segmented Characters

![out](https://user-images.githubusercontent.com/19779081/45310754-cefeaa80-b544-11e8-9e17-be92cb7eb183.png)

>Predicted Value

![out](https://user-images.githubusercontent.com/19779081/45310789-e2aa1100-b544-11e8-94cd-a5bbdc2e7888.PNG)


2. Car Image


>Original Image

![car](https://user-images.githubusercontent.com/19779081/45311273-38cb8400-b546-11e8-9cb0-a660bf07806e.png)

>License Plate Detected

![out7](https://user-images.githubusercontent.com/19779081/45311348-73352100-b546-11e8-9cd9-89f755690772.png)

>Segmented characters

![out8](https://user-images.githubusercontent.com/19779081/45311349-73352100-b546-11e8-89f2-6c2f3df9dc1f.png)

>Predicted characters

![out9](https://user-images.githubusercontent.com/19779081/45311365-7d571f80-b546-11e8-9775-a44ecc7213c7.PNG)
