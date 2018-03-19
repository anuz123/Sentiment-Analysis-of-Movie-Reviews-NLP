# Naive-Bayes Algorithm for text mining 

# Load required libraries

library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)

# Reading the Data
setwd("D:/machine_learning")
df<- read.csv("movie_reviews.csv", stringsAsFactors = FALSE)
glimpse(df)

# Randomize the dataset
set.seed(123)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
glimpse(df)

# Convert the 'class' variable from character to factor.
df$class <- as.factor(df$class)

# Bag of Words Tokenisation
corpus <- Corpus(VectorSource(df$text))
# Inspect the corpus
corpus
inspect(corpus[1:3])

# Data Cleanup

corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

# Matrix representation of Bag of Words : The Document Term Matrix
# The rows of the DTM correspond to documents in the collection, columns correspond to terms, and its elements are the term frequencies. 

dtm <- DocumentTermMatrix(corpus.clean)
# Inspect the dtm
inspect(dtm[40:50, 10:15])

# Partitioning the Data
# Next, we create 75:25 partitions of the dataframe, corpus and document term matrix for training and testing purposes.

df.train <- df[1:1500,]
df.test <- df[1501:2000,]

dtm.train <- dtm[1:1500,]
dtm.test <- dtm[1501:2000,]

corpus.clean.train <- corpus.clean[1:1500]
corpus.clean.test <- corpus.clean[1501:2000]

# Feature Selection
dim(dtm.train)  

# Restrict the DTM to use only the frequent words using the 'dictionary' option.
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))

# Use only 5 most frequent words (fivefreq) to build the DTM

dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))

dim(dtm.train.nb)

dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))

dim(dtm.train.nb)

# The Naive Bayes algorithm
# Term frequencies are replaced by Boolean presence/absence features
convert_count <- function(x) 
  {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

# Training the Naive Bayes Model
# Train the classifier
classifier <- naiveBayes(trainNB, df.train$class, laplace = 1)
# Testing the Predictions
system.time(pred <- predict(classifier, newdata=testNB))
table("Predictions"= pred,  "Actual" = df.test$class )
# Confusion Matrix
conf.mat <- confusionMatrix(pred, df.test$class)
conf.mat
conf.mat$byClass
conf.mat$overall
# Prediction Accuracy
conf.mat$overall['Accuracy']   # 78.6% Accuracy
