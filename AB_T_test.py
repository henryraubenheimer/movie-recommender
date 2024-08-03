"""
Perform a T-test on the feedback of two groups of users
"""
import csv

import numpy as np

feedbackA, feedbackB = [], []  # the feedback of users in A and B

with open('feedback.csv', 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader, None)
    for row in csv_reader:
        user_id, movie_id, feedback, versionid = row
        if versionid == 'A':
            feedbackA.append(float(feedback))
        else:
            feedbackB.append(float(feedback))

t = np.mean(feedbackA)-np.mean(feedbackB)
t = t/np.sqrt(np.var(feedbackA)/len(feedbackA))
t += np.var(feedbackB)/len(feedbackB)
print('A Z value of '+str(t)+' is produced by the T-test.')
