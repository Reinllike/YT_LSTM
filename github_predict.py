#!/usr/bin/env python
# coding: utf-8

# In[10]:


#connect youtube api

import requests
import csv
from datetime import datetime
class YouTubeCrawler():
    # Each delivery request requires an API Key.
    def __init__(self, api_key):
        self.base_url = "https://www.googleapis.com/youtube/v3/"
        self.api_key = api_key

    # The returned data should be converted to JSON, including the functions used for combining URLs and requests, and also for request failure processing.
    def get_html_to_json(self, path):
        """Combine URL and GET pages and convert to JSON"""
        api_url = f"{self.base_url}{path}&key={self.api_key}"
        r = requests.get(api_url)
        if r.status_code == requests.codes.ok:
            data = r.json()
        else:
            data = None
        return data

    # retrun number of subscribers
    def getsub_count(self, channel_id, part='snippet,statistics'):
        """Get the video list ID of the channel upload"""
        # UC7ia-A8gma8qcdC6GDcjwsQ
        path = f'channels?part={part}&id={channel_id}'
        data = self.get_html_to_json(path)
        try:
            subCount = data['items'][0]["statistics"]["subscriberCount"]
        except KeyError:
            subCount = None
        return subCount

    # return video information
    def get_video(self, video_id, part='snippet,statistics'):
        """get video information"""
        path = f'videos?part={part}&id={video_id}'
        data = self.get_html_to_json(path)
        if not data:
            return {}

        # The following code is to process the whole data and extract the part we need
        data_item = data['items'][0]

        try:
            # 2019-09-29T04:17:05Z
            time_ = datetime.strptime(data_item['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            # wrong date format
            time_ = None

        url_ = f"https://www.youtube.com/watch?v={data_item['id']}"

        try:
            comment_ = data_item['statistics']['commentCount']
        except:
            # Message deactivation
            comment_ = 0

        try:
            like_ = data_item['statistics']['likeCount']
        except:
            # Message Likes
            like_ = 0
        info = {
            'id': data_item['id'],                                          # video ID
            'channelTitle': data_item['snippet']['channelTitle'],           # channel name
            'publishedAt': time_,                                           # video release time
            'video_url': url_,                                              # video url
            'title': data_item['snippet']['title'],                         # video title
            # 'description': data_item['snippet']['description'],           # video description
            'likeCount': like_,                                             # likes
            # 'dislikeCount': data_item['statistics']['dislikeCount'],      # unlikes
            'commentCount': comment_,                                       # Number of Comments
            'viewCount': data_item['statistics']['viewCount']               # views
        }

        return info


# In[11]:


# Key in the target youtuber channel's id and the target video id，the code will crawl the data we need.
# Get the Channel ID: https://commentpicker.com/youtube-channel-id.php

# youtube_channel_id = "Target youtuber channel's id"
# video_id = 'Target video id'
youtube_channel_id = "UCIF_gt4BfsWyM_2GOcKXyEQ"
video_id = 'besQJCZ20j4'

# youtube_spider = YouTubeCrawler("youtube api key") #input your own youtube api key
youtube_spider = YouTubeCrawler("AIzaSyBnL0UuXW95x87qL43PEaBh779G7UWODEk")

channel_subCount = youtube_spider.getsub_count(youtube_channel_id)
video_info = youtube_spider.get_video(video_id)
video_info['subscriberCount'] = channel_subCount

predict_Title_ = video_info['title']
predict_PubTime = video_info['publishedAt']
predict_Subscriber = video_info['subscriberCount']
viewCount = video_info['viewCount']

print("頻道名稱 ：" + video_info['channelTitle'])
print("影片標題 ：" + predict_Title_)
print("發佈時間 ：" + str(predict_PubTime))
print("頻道訂閱 ：" + predict_Subscriber)


# In[12]:


#Prediction

import opencc
converter = opencc.OpenCC('s2t.json')
# Step 0: key in youtube data
predict_Title = []
word = converter.convert(predict_Title_)  #turn Simplified Chinese into Traditional Chinese
predict_Title.append(word)

#import title
sentence_list = predict_Title
print("Sentence_list : " , sentence_list)

# Step 1: word segmentation
from ckiptagger import data_utils, construct_dictionary, WS
ws = WS("./data")
word_to_weight = {
    "老高與小茉 Mr & Mrs Gao": 1,
}

dictionary = construct_dictionary(word_to_weight)

word_sentence_list = ws(
    sentence_list,
    sentence_segmentation=True,  #To consider delimiters
    segment_delimiter_set={",", "。", ":", "?", "!", ";", "【", "】", "|"},  # This is the defualt set of delimiters
    coerce_dictionary=dictionary,  # words in this dictionary are forced
)
print("Origin word_sentence_list : ", word_sentence_list)
del ws

# Step 2: import pre_trained Word2Vec model
import gensim
from gensim.models.word2vec import Word2Vec
model = gensim.models.KeyedVectors.load_word2vec_format('pre_trained.bin',
                                                        unicode_errors='ignore',
                                                        binary=True)

# Step 3: remove meaningless words
words = []
word_rm = []

word_sentence_list_copy = word_sentence_list.copy()
for words_list in word_sentence_list_copy:
    if len(words_list) < 2:    # To deal with titles that are too short (less than 2 hyphenated words), add the word "video"
        words_list.append("影片")
        
    words_list_copy = words_list.copy()
    for e in words_list_copy:
        if len(e) < 2:          # remove conjunction
            words_list.remove(e)
            word_rm.append(e)
        elif len(e) > 6:        # remove the words that are too long
            words_list.remove(e)
            word_rm.append(e)
        elif e[0] == ' ' or e[-1] == ' ':
            words_list.remove(e)
            word_rm.append(e)


# Step 4: remove the word that don't exist in the word2vector pre-trained model
word_notInModel = []
for words_list in word_sentence_list:
    words_list_copy = words_list.copy()
    for w in words_list_copy:
        try:  
            model[w]
        except:  
            words_list.remove(w)
            word_notInModel.append(w)

#After processing the data, save it in word_sentence_list, and go to the next step

print("After word_sentence_list : " , word_sentence_list)

#turn the title into word2vector and save in docu_array
import numpy as np 
zero_array = np.zeros(100)  

for i in range(0 , len(word_sentence_list)):
    try:
        if i == 0:
            docu_array = np.mean(model[word_sentence_list[i]] , axis=0)
        else:
            docu_array = np.vstack((docu_array , np.mean(model[word_sentence_list[i]] , axis=0) ))
        
    except KeyError as e:
        print(repr(e))
        
#reshape the docu_array to the data structure that we can use
docu2=docu_array.reshape((1,1,100))

import statistics
import time
import numpy as np

predict_PubTime = str(predict_PubTime)
predict_Subscriber = int(predict_Subscriber)

times = predict_PubTime # The format of time is str
sub = predict_Subscriber

#When we deal with the data that for training model, i had standardized the time, so when i want to predict new target, the time of the prediction should also be standardized.
#statistics.stdev(timelist)=55128744 statistics.mean(timelist)=89176696
#Time
std=55128744
mean=89176696 

#process time's value into Numerical value
struct_time = time.strptime("2022-08-05 20:00:00", "%Y-%m-%d %H:%M:%S") 
time_stop = int(time.mktime(struct_time)) 
time_ord = time.strptime(times, "%Y-%m-%d %H:%M:%S")
time_now = int(time.mktime(time_ord))
timetonow = time_stop - time_now

time_arr = np.array((timetonow-mean)*10/std) 
timedata=time_arr.reshape((1,1,1))

x_label=np.append(docu2, timedata, axis = 2 )

#subscriber
#When we deal with the subscribers data for training model, we let the number of subscribers divide by 100000, to make the word2vector, time, and subscriber data could in a similar range of values.
sub /= 100000
subarray=np.array(sub)
subdata=subarray.reshape((1,1,1))
x_label=np.append(x_label, subdata, axis = 2 )


from tensorflow.keras.models import load_model
model = load_model('.\\model\\second_trash_model.h5') #load model
prediction = model.predict(x_label) * 100000

#Output
delV = abs(int(viewCount)-int(prediction))
print()
print("頻道名稱 ：" + video_info['channelTitle'])
print("影片標題 ：" + predict_Title_)
print("發佈時間 ：" + str(predict_PubTime))
print("頻道訂閱 ：" + str(predict_Subscriber))
print("真實點閱 : " + str(int(viewCount)))
print("預測點閱 : " + str(int(prediction)))
print("Accuracy : {:4.1f}%".format( ((1-( delV / max(int(prediction), int(viewCount)) ))*100) )) #the Accuracy is difined by ous to observe the results simply

