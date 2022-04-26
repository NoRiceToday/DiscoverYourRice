# DiscoverYourRice
A context-sensitive WebApp, that uses machine learning to customize user experience. More specifically, it autoplays an embedded youtube video if the user is sitting or standing and stops it, if the user is walking.

- Uses a Random Forest Classifier to determine whether the user is sitting, standing or walking.
  - K-NearestNeighbors, SupportVectorClassification, DecisionTree and NaiveBayes Models have all been trained and tested as well. 
  - Models have been compared using a 10-fold-cross validation.
  - The Random Forest achieved the best accuracy for our problem.
  - Training was done on a Nvidia Geforce GTX 1070.
  - Model has been ported to a JavaScript file using Porter ("Classifier.js").
- Data for Training was taken from a private influx db, downloaded to a csv and then wrangled to fit training requirements.
  - Students anonymously recorded data samples of them using their phone in corresponding positions.
  -  Due to the sheer size of the data, it has not been uploaded to this repository.
- WebApp has been developed using Angular and Bootstrap and is kept fairly minimal for illustration purposes.

Screenshot of the WebApp:
![msedge_xSaL6jVEhe](https://user-images.githubusercontent.com/25351150/165299269-4de10a17-15c8-4f43-84c3-1621939dd76f.jpg)
