# YouTube Bi-LSTM Model

## Use Deep learning to predict the viewcounts of YouTube videos
We developed a Bi-LSTM Model to study the relationship between the title and the viewcounts of YouTube videos. This model allows YouTubers to predict the viewcounts of their own video titles before the video is published and pick the one with the best viewcounts performance!


## Usage
### Model Training
```
# Get data from Youtube via Youtube_Crawler.py
$ python Youtube_Crawler.py  
$ python model_F.py
```


### Visualization
`$ python _Word2Vec_visualization_3D.py`  
![word2Vec_gif](https://user-images.githubusercontent.com/111637364/186734029-2d3c3d5e-e059-4a75-82d3-3ac3eb5242c7.gif)

`$ python _Doc2Vec_visualization.py`  
![doc2Vec](https://user-images.githubusercontent.com/111637364/186747996-65ea93cc-5dc1-452b-8874-51aec3158ffe.jpg)


### Prediction
`$ python github_predict.py`  

