# -*- coding: utf-8 -*-
# add the previous line for arabic encodeing

'''Dataset source: Abdulla N. A., Mahyoub N. A., Shehab M., Al-Ayyoub M.,
        ìArabic Sentiment Analysis: Corpus-based and Lexicon-basedî,
        IEEE conference on Applied Electrical Engineering and Computing Technologies (AEECT 2013),
        December 3-12, 2013, Amman, Jordan. (Accepted for Publication).'''

# creating Naive Bayes Classifier
from textblob.classifiers import NaiveBayesClassifier

cl = NaiveBayesClassifier("train.csv", format="csv")
#cl = NaiveBayesClassifier(train)

# Test model with its two labels
print cl.classify(u" احسن علاج هذا")

# second cl model test
prob_dist = cl.prob_classify(u"ك يوم يا ظالم,")
print prob_dist.max()
print prob_dist.prob("positive")
print prob_dist.prob("negative")

# compute the accuracy on our test set
print "accuracy on the test set:{} ".format(cl.accuracy("testing.csv", format="csv"))

# display a listing of the most informative features.
cl.show_informative_features(5)

# add new data
new_data = [(u"كلام صحيح من شان هيك الدول اللي ما فيها بطالة والمجتمعات المفتوحة بتقل فيها المشاكل النفسية", 'positive'),
           (u"لا طبعا التقرب الى الله هو خير علاج للحالات النفسية", 'positive'),
           (u"تفائلوا بالخير تجدوه", 'positive'),
           (u"يا ترى الحكومه بدها تزيد دعم المواطن الي الله يكون في عونه", 'negative')]

# updating classifiers with new data
cl.update(new_data)

# test accuracy after adding new data to the generated model
print "accuracy on the test set:{} ".format(cl.accuracy("testing.csv", format="csv"))



