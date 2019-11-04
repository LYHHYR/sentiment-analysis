# Sentiment Analysis


### 1.
Twitter mood predicts the stock market
https://arxiv.org/pdf/1010.3003.pdf

### 2. TextBlob


```python
text = "I am happy today. I feel sad today."
from textblob import TextBlob
blob = TextBlob(text)
blob
```


```python
blob.sentences
```




    [Sentence("I am happy today."), Sentence("I feel sad today.")]




```python
blob.sentences[0].sentiment
```




    Sentiment(polarity=0.8, subjectivity=1.0)



polarity:[negative,positive]=[-1,1]


```python
blob.sentences[1].sentiment
```




    Sentiment(polarity=-0.5, subjectivity=1.0)




```python
blob.sentiment
```




    Sentiment(polarity=0.15000000000000002, subjectivity=1.0)



### 3. SnowNLP


```python
text = u"我今天很快乐。我今天很愤怒。"
from snownlp import SnowNLP
s = SnowNLP(text)
```


```python
for sentence in s.sentences:
    print(sentence)
```

    我今天很快乐
    我今天很愤怒



```python
s1 = SnowNLP(s.sentences[0])
s1.sentiments
```




    0.971889316039116




```python
s2 = SnowNLP(s.sentences[1])
s2.sentiments
```




    0.07763913772213482


