# 2024 AI CITY CHALLENGE 

## Track 5: Detecting Violation of Helmet Rule for Motorcyclists

Motorcycles are one of the most popular modes of transportation, particularly in developing countries such as India. Due to lesser protection compared to cars and other standard vehicles, motorcycle riders are exposed to a greater risk of crashes. Therefore, wearing helmets for motorcycle riders is mandatory as per traffic rules and automatic detection of motorcyclists without helmets is one of the critical tasks to enforce strict regulatory traffic safety measures.

- Data

The training dataset contains 100 videos and groundtruth bounding boxes of motorcycle and motorcycle rider(s) with or without helmets. Each video is 20 seconds duration, recorded at 10 fps. The video resolution is 1920×1080.

Each motorcycle in the annotated frame has bounding box annotation of each rider with or without helmet information, for upto a maximum of 4 riders in a motorcycle. The class id (labels) of the object classes in this dataset is as follows:
