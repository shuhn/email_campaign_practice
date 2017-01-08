# Conclusions

## Q1
What percentage of users opened the email and what percentage clicked on the link within the email?

Open Rate | 10.345 %

CTR:      | 2.119 %

## Q2
The VP of marketing thinks that it is stupid to send emails to a random subset and in a random way. Based on all the information you have about the emails that were sent, can you build a model to optimize in future email campaigns to maximize the probability of users clicking on the link inside the email?

I built a variety of models that sought to predict whether or not a given user would click on the link inside the email (CTR). I made several model decisions based on the distributions of CTRs. I would like to highlight three main variables: DayofWeek, PastPurchases, Hour.

A) DayofWeek

There appears to be three main groups of CTRs:
- Monday / Tuesday / Wednesday / Thursday
- Saturday / Sunday  
- Friday

I split this variable into three categorical bins to capture this information.

![CTR_by_Day](/assets/Snip20170107_19.png)

B) Past Purchases

Repeat customers are more likely to click through than new customers. In the future it may make sense to optimize two models; one for new customers and one for return customers. It would be interesting to explore if the behaviors of these two groups are different.


![CTR_by_Day](/assets/Snip20170107_20.png)

C) Hour

The volume at each hour varied greatly. Most emails were sent out in the morning hours. I decided to bucket the hours into four sections:
- 1 - 6
- 7 - 12
- 13 - 18
- 19 - 24

![CTR_by_Day](/assets/Snip20170107_21.png)

I tuned a random forest classifier and logistic regression on the transformed dataset. Both models performed significantly better on the training data (and cross validated data) than the testing data, suggesting some degree of overfit. **My goal was to determine the features most important to a user's CTR.** Although the precision scores for both models are low (~3.8%), the accuracy and recall were acceptably high. The low precision suggests that the model had a high FPR. This was an acceptance tradeoff given the goal of this test.

 Graphing the feature importances of the random forest classifier, it can be seen that:
 - is_personalized
 - is_short
 - is_MTWTh
 - hour_7_12

 Are the four most important variable that we as a company have control over. Targeting these variables in future email campaigns would maximize our CTR.

![CTR_by_Day](/assets/Snip20170107_22.png)

## Q3
By how much do you think your model would improve click through rate ( defined as # of users who click on the link / total users who received the email). How would you test that?

I use a beta distribution to simulate the change in CTR that we could expect from using the optimized model. I split all of our recorded data into two groups:
- is_personalized, is_short, is_MTWTh, hour_7_12
- all else

optimum	| clicked |	counter	| ctr
------- | ------- | ------- | ---
NO |	1851.0	| 93289 | 	1.9842%
YES |	268.0 | 	6711	| 3.9934%

This bayesian approach allows me to conclude that there is a 98% chance that optimizing our email campaign would lead to a 100% lift in our CTR (from 1.98% to 4.0%). There is reason to be skeptical of this test due to the relatively small sample size of the optimized group. **However, it is justifiable to conclude that optimizing emails to fit these characteristics would substantially increase our CTR.**

# Q4
Did you find any interesting pattern on how the email campaign performed for different segments of users? Explain.

As I mentioned earlier, it's clear that return customers have higher CTR's than new customers. We should explore ways of targeting these two subsets of users differently. In the future I would like to explore the differences in behaviors between these two groups.

It also appears that CTR's vary by country. It would be interesting to explore how we tailor our emails based on the country of the user. Do we adapt the language? Do we record the time sent as the user's local time or our system's time? These would be interesting questions to explore with more time.
