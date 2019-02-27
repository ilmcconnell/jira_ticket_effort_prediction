# JIRA Ticket Effort Prediciton

## Rationale:
* predict whether a particular JIRA ticket will be easy or hard to resolve. The Easy class is composed of all tickets that took 1 hour or less of work to complete, the Hard class is everything else.
* The goal is to automatically apply this to assignment to queued tickets, expediting their assignment a particular sprint or queue - eliminating a regular admin task

## Approach
* Get historical work data in JIRA for tickets, classify them into easy and hard tickets.
* TFIDF transform ticket descriptions and use the TFIDF output as features for a Naive Bayes classifier
* The two classes predicted 0 = Easy and 1 = Hard

### Confusion Matrix
|                  |    predicted   0    |    predicted   1    |    sum    |   
|------------------|:-------------------:|:-------------------:|:---------:|
|    actual   0    |    70               |    106              |    176    |
|    actual   1    |    11               |    354              |    365    |
|    sum           |    81               |    460              |    541    | 

### Classifier Evaluation Statistics 
|   statistic                  |  value      |
|------------------------------|:-----------:|
|    null accuracy*            |   0.66      |
|    accuracy                  |    0.78     |  
|    recall/true   positive    |    0.97     |  
|    false   positive rate     |    0.60     |   
|    True   negative rate      |    0.40     |  
|    False   negative rate     |    0.03     |  
|    precision                 |    0.77     |
*null accuraccy is not a model output but simply the result of 'predicting' a ticket class by always reporting 'hard'

## Summary
* Despite obvious shortcomings this model is presently useful for the stated goal. 
* If the model is applied as it stands, the false positive rate would put a lot of easy tickets in the hard sprint, which is OK, and very few hard tickets will be assigned into the easy queue.

## Next Steps
* Improve the model
  * get more features
  * balance classes in the data and rerun classifier
* Address false positive rate (incorporate a loss function)
* Potential to use regression not just classification	
  * potentially Estimate workload for an entire sprint.	
