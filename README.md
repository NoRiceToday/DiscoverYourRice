# DiscoverYourRice
A context-sensitive WebApp, that uses machine learning to customize user experience. More specifically, it autoplays an embedded youtube video if the user is sitting or standing and stops it, if the user is walking. This is done using real-time classification of mobile sensor data.

![msedge_xSaL6jVEhe](https://user-images.githubusercontent.com/25351150/165299269-4de10a17-15c8-4f43-84c3-1621939dd76f.jpg)

## ML Model
Uses a Random Forest Classifier to determine whether the user is sitting, standing or walking.
- Multilayer Perceptron, K-Nearest Neighbors, Support Vector Classification/Machine, Decision Tree and Naive Bayes Models have all been trained and tested as well. 
- Models have been compared using a 10-fold-cross validation.
- The Random Forest achieved the best accuracy for our problem.

## Data
Data for Training was taken from a private influx db, downloaded to a csv and then wrangled to fit training requirements.
- Students anonymously recorded data samples of them using their phone in corresponding positions.
- Due to privacy concerns, it has not been uploaded to this repository.

## Training
- The classifier uses multiple acceleration and tilting sensors of the mobile device to determine the label.
- Training on the sensor data directly is too noisy. So instead, we trained the model using the mean and variance.
  - 1 second and 2 second bucket/window sizes have been tested and compared. 
  - 2 second window not only increased latency of the final WebApp but also introduced more deviation.
  - Thus, 1 second windows have been used.
- Training was done on a Nvidia Geforce GTX 1070.

## WebApp
WebApp has been developed using Angular and Bootstrap and is kept fairly minimal, as it is mainly for illustration purposes.
- Model has been ported to a JavaScript file using Porter to use in realtime ("Classifier.js").


## Usage Instructions
If you only want to use the WebApp:
1. Download the 'webApp' folder
2. Execute "npm install", "ng update" and "npm update" in '/webApp' as well as '/webApp/KontextSensitiveApp'.
3. Execute 'ng serve' from the '/webApp/KontextSensitiveApp' directory.
4. Open 'localhost:4200' in your browser.
5. Enjoy! It should work now.

*Note: The classification will always classify the state as sitting when using it in a desktop environment. To test the real-time classification feature, you need to open it on a mobile device.*

To play around with the classifier or data wrangling, you probably need to refactor quite a few variables as this project has been developed only to be used by me for a master's lecture. Edit at your own risk!


