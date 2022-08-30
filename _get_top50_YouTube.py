import requests
import pprint
def getTop50ID(YOUTUBE_API_KEY):
    base_url = "https://www.googleapis.com/youtube/v3/"
    api_function = "videos"
    query = {
        "key": YOUTUBE_API_KEY,
        "chart": "mostPopular",
        "maxResults": 50,
        "regionCode": "TW",
        "part": "snippet,contentDetails,statistics",
    }
    response = requests.get(base_url + api_function, params=query)

    if response.status_code == 200:
        response_dict = response.json()
        # Get the "Channel Title" of the Top50 YouTuber in TW
        channelTitle_ = {
            i + 1: response_dict["items"][i]["snippet"]["channelTitle"]
            for i in range(0, 50)
        }
        # Get the "View Count" of the Top50 YouTuber in TW
        viewCount_ = {
            i + 1: response_dict["items"][i]["statistics"]["viewCount"]
            for i in range(0, 50)
        }
        # Get the "Channel Id" of the Top50 YouTuber in TW
        channelId_ = {
            i + 1: response_dict["items"][i]["snippet"]["channelId"]
            for i in range(0, 50)
        }

    pprint.pprint(channelTitle_)                # channelTitle
    pprint.pprint(viewCount_)                   # viewCount
    pprint.pprint(channelId_)                   # channelId
    listOfTop50 = (list(channelId_.values()))
    # return the "Channel ID" of the Top50 YouTuber for "Youtube_Crawler"
    return listOfTop50
#
# print(len(topForTW()))
# print((topForTW()))