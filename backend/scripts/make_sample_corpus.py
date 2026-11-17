"""Builds data/sample_corpus.json: a SYNTHETIC corpus of educational video transcripts.

Five fictional YouTube educators each teach a subset of 18 machine-learning topics. Transcripts are
assembled from topic sentence pools, so the corpus has realistic structure (prerequisites mentioned
before advanced ideas, overlapping coverage between educators) without using anyone's real content.
Run:  python scripts/make_sample_corpus.py
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

TOPICS = {
    "Python basics": [
        "Python is a general purpose programming language that is popular in data science.",
        "A variable stores a value, and every value in Python has a data type such as integer, float or string.",
        "We use lists to store an ordered collection of values and dictionaries to store key value pairs.",
        "A for loop repeats a block of code for every item in a list.",
        "Functions let us reuse code, so we define a function once with def and call it many times.",
        "Indentation matters in Python because it defines which statements belong to a loop or a function.",
        "An if statement lets the program make decisions based on a condition.",
        "Libraries like numpy and pandas extend Python with tools for numerical computing and data analysis.",
    ],
    "NumPy arrays": [
        "NumPy gives us the array, a fast container for numbers that is much quicker than a Python list.",
        "Vectorized operations apply a calculation to every element of an array without writing a for loop.",
        "Broadcasting lets numpy combine arrays of different shapes in a single operation.",
        "The shape of an array tells us how many rows and columns it has, and reshape changes it.",
        "Matrix multiplication with numpy is the foundation of linear algebra in machine learning.",
        "Indexing and slicing let us select parts of an array, just like with Python lists.",
        "Random number generation in numpy is useful for simulations and for initializing model weights.",
    ],
    "Pandas dataframes": [
        "A pandas dataframe is a table of rows and columns, similar to a spreadsheet.",
        "We load a CSV file into a dataframe with read_csv and inspect it with head and describe.",
        "Missing values are common in real data, and pandas lets us drop them or fill them in.",
        "Groupby splits the data into groups, applies a function such as the mean, and combines the results.",
        "Filtering rows with a boolean condition is one of the most common pandas operations.",
        "Each column of a dataframe is built on a numpy array, so vectorized operations are fast.",
        "Data cleaning with pandas usually takes more time than building the machine learning model.",
    ],
    "Data visualization": [
        "Data visualization helps us understand a dataset before we build any model.",
        "A histogram shows the distribution of a single variable.",
        "A scatter plot shows the relationship between two variables and can reveal correlation.",
        "Matplotlib and seaborn are the most common plotting libraries in Python.",
        "A box plot summarises the median, the quartiles and the outliers of a variable.",
        "Good charts have clear labels, a sensible scale and as little clutter as possible.",
        "We often plot a pandas dataframe directly to explore the data quickly.",
    ],
    "Descriptive statistics": [
        "The mean is the average value, while the median is the middle value when the data is sorted.",
        "Variance and standard deviation measure how spread out the data is around the mean.",
        "A distribution describes how often each value occurs in the data.",
        "The normal distribution is the bell shaped curve that appears again and again in statistics.",
        "Outliers can pull the mean far away from the median, so the median is more robust.",
        "Correlation measures how strongly two variables move together, from minus one to plus one.",
        "Percentiles and quartiles tell us where a value sits within a distribution.",
    ],
    "Probability": [
        "Probability measures how likely an event is, on a scale from zero to one.",
        "Conditional probability is the probability of an event given that another event has happened.",
        "Bayes theorem lets us update a probability when we observe new evidence.",
        "A random variable maps outcomes of an experiment to numbers, and it has a distribution.",
        "Independent events do not influence each other, so their probabilities multiply.",
        "Expected value is the long run average of a random variable.",
        "Probability is the language that machine learning models use to express uncertainty.",
    ],
    "Linear regression": [
        "Linear regression fits a straight line that predicts a continuous target from input features.",
        "The model learns a weight for each feature and a bias term, also called the intercept.",
        "We measure the error with the mean squared error, the average of the squared residuals.",
        "The residual is the difference between the actual value and the predicted value.",
        "Linear regression assumes a roughly linear relationship, which a scatter plot can check.",
        "The best weights can be found with the normal equation or with gradient descent.",
        "R squared tells us how much of the variance in the target the model explains.",
    ],
    "Gradient descent": [
        "Gradient descent is an optimization algorithm that minimizes a loss function step by step.",
        "The gradient points in the direction of steepest increase, so we move in the opposite direction.",
        "The learning rate controls the size of each step, and choosing it well is important.",
        "If the learning rate is too large the loss can diverge, and if it is too small training is slow.",
        "Stochastic gradient descent uses a small batch of data for each update, which is faster.",
        "Gradient descent trains linear regression, logistic regression and neural networks alike.",
        "A loss curve that goes down smoothly tells us that the optimization is working.",
    ],
    "Logistic regression": [
        "Logistic regression is a classification model that predicts the probability of a class.",
        "The sigmoid function squashes any number into a probability between zero and one.",
        "We train logistic regression by minimizing the cross entropy loss with gradient descent.",
        "A decision threshold, often 0.5, turns the predicted probability into a class label.",
        "Despite its name, logistic regression is used for classification and not for regression.",
        "The weights of logistic regression can be interpreted as changes in the log odds.",
        "Logistic regression is a strong baseline before trying more complex classifiers.",
    ],
    "Overfitting and regularization": [
        "Overfitting happens when a model memorizes the training data and fails on new data.",
        "The bias variance tradeoff explains why simple models underfit and complex models overfit.",
        "Regularization adds a penalty on large weights to keep the model simpler.",
        "L1 regularization, also called lasso, can push some weights exactly to zero.",
        "L2 regularization, also called ridge, shrinks all weights smoothly towards zero.",
        "We detect overfitting by comparing the training error with the validation error.",
        "More data, simpler models and early stopping all help reduce overfitting.",
    ],
    "Model evaluation": [
        "We split the data into a training set and a test set to estimate performance on new data.",
        "Cross validation trains and tests the model several times on different folds of the data.",
        "Accuracy can be misleading when the classes are imbalanced.",
        "Precision tells us how many predicted positives are correct, and recall tells us how many positives we found.",
        "The confusion matrix counts true positives, false positives, true negatives and false negatives.",
        "The ROC curve and the area under it summarise performance across all decision thresholds.",
        "The F1 score is the harmonic mean of precision and recall.",
    ],
    "Decision trees and random forests": [
        "A decision tree splits the data with a sequence of yes or no questions about the features.",
        "Each split is chosen to make the groups purer, measured with gini impurity or entropy.",
        "Deep decision trees overfit easily, so we limit their depth.",
        "A random forest trains many decision trees on random samples of the data and averages them.",
        "Averaging many trees reduces variance, which is why random forests overfit less.",
        "Feature importance from a random forest shows which features the model relies on.",
        "Gradient boosting builds trees one after another, each correcting the errors of the previous ones.",
    ],
    "K-means clustering": [
        "Clustering is unsupervised learning, because there are no labels to predict.",
        "K-means assigns every point to the nearest of k centroids and then moves each centroid to the mean of its points.",
        "We repeat the assignment and update steps until the centroids stop moving.",
        "Choosing k is hard, and the elbow method and the silhouette score can help.",
        "K-means uses euclidean distance, so features should be scaled first.",
        "Clustering is used for customer segmentation and for grouping similar documents.",
        "K-means can get stuck in a poor solution, so we run it several times with different starts.",
    ],
    "Principal component analysis": [
        "Principal component analysis, or PCA, reduces the number of dimensions while keeping most of the variance.",
        "PCA finds new axes called principal components that are combinations of the original features.",
        "The first principal component points in the direction of the largest variance.",
        "PCA relies on linear algebra, specifically the eigenvectors of the covariance matrix.",
        "We standardize features before PCA so that large scale features do not dominate.",
        "PCA is useful for visualizing high dimensional data in two dimensions.",
        "The explained variance ratio tells us how much information each component keeps.",
    ],
    "Neural networks": [
        "A neural network is made of layers of neurons, and each neuron computes a weighted sum followed by an activation function.",
        "Activation functions such as relu and sigmoid let the network learn non linear patterns.",
        "A network with hidden layers can approximate very complex functions.",
        "We train neural networks with gradient descent by minimizing a loss function.",
        "Logistic regression is like a neural network with a single neuron and a sigmoid activation.",
        "Neural networks need a lot of data and are prone to overfitting, so regularization such as dropout helps.",
        "Frameworks like PyTorch and TensorFlow make building neural networks easier.",
    ],
    "Backpropagation": [
        "Backpropagation computes the gradient of the loss with respect to every weight in a neural network.",
        "It applies the chain rule from calculus, layer by layer, from the output back to the input.",
        "The forward pass computes predictions, and the backward pass computes gradients.",
        "Once we have the gradients, gradient descent updates the weights.",
        "Vanishing gradients make deep networks hard to train with sigmoid activations.",
        "Relu activations and careful weight initialization help gradients flow through deep networks.",
        "Automatic differentiation in modern frameworks performs backpropagation for us.",
    ],
    "Convolutional neural networks": [
        "Convolutional neural networks, or CNNs, are neural networks designed for images.",
        "A convolution slides a small filter over the image to detect patterns such as edges.",
        "Pooling layers reduce the size of the feature maps and make the network more robust.",
        "Early layers detect simple edges while deeper layers detect shapes and objects.",
        "CNNs share weights across the image, so they need far fewer parameters than a fully connected network.",
        "Transfer learning reuses a CNN pretrained on a large dataset for a new task.",
        "CNNs are trained with backpropagation and gradient descent like any neural network.",
    ],
    "Transformers and attention": [
        "Transformers are neural networks built around the attention mechanism.",
        "Attention lets every word in a sentence look at every other word and decide what is relevant.",
        "Self attention computes queries, keys and values with matrix multiplication.",
        "Positional encoding tells the transformer the order of the words.",
        "Large language models such as GPT are transformers trained on huge amounts of text.",
        "Word embeddings represent words as vectors so that similar words are close together.",
        "Transformers replaced recurrent neural networks for most language tasks because they train in parallel.",
    ],
}
ORDER = list(TOPICS)  # curriculum order, beginner -> advanced

EDUCATORS = {
    "Data Dojo": {"topics": ORDER[0:9], "intro": "Hey everyone, welcome back to Data Dojo.", "filler": "Alright, so",
                  "outro": "If this helped, practise the code yourself and see you in the next lesson."},
    "Learn with Meera": {"topics": ORDER[0:6] + ORDER[10:14], "intro": "Hello and welcome, I am Meera.", "filler": "Now,",
                         "outro": "Try the exercise in the description and let me know your questions."},
    "The Math Lab": {"topics": ORDER[4:8] + ORDER[13:16], "intro": "Welcome to The Math Lab.", "filler": "Mathematically speaking,",
                     "outro": "Next time we will build on this result."},
    "Code and Coffee": {"topics": [ORDER[1], ORDER[2], ORDER[6], ORDER[8], ORDER[10], ORDER[11], ORDER[12], ORDER[14]],
                        "intro": "Grab a coffee, today we are coding.", "filler": "Okay so basically",
                        "outro": "The notebook is linked below, happy coding."},
    "AI Simplified": {"topics": ORDER[9:18], "intro": "Welcome to AI Simplified.", "filler": "In simple words,",
                      "outro": "Subscribe for the next part of the deep learning series."},
}


def build(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    videos = []
    start = date(2023, 1, 9)
    for e_i, (edu, cfg) in enumerate(EDUCATORS.items()):
        day = start + timedelta(days=rng.randint(0, 60) + e_i * 25)
        for t_i, topic in enumerate(cfg["topics"]):
            sentences = TOPICS[topic][:]
            rng.shuffle(sentences)
            body = sentences[:6]
            prereq = ORDER.index(topic) - 1
            recap = []
            if prereq >= 0:
                p = ORDER[prereq]
                recap.append(f"Before we start, remember what we covered about {p.lower()}. {TOPICS[p][0]}")
            lines = [cfg["intro"], f"In this video we look at {topic.lower()}."] + recap
            for i, s in enumerate(body):
                lines.append((f"{cfg['filler']} " + s[0].lower() + s[1:]) if i % 3 == 1 else s)
            lines += [f"So that is a quick summary of {topic.lower()}.", cfg["outro"]]
            vid = f"{edu.split()[0].lower()}-{t_i + 1:02d}"
            videos.append({"video_id": vid, "educator": edu, "title": topic,
                           "published": day.isoformat(), "views": int(rng.lognormvariate(9.2, 0.8)),
                           "url": "", "synthetic": True, "transcript": " ".join(lines)})
            day += timedelta(days=rng.randint(10, 30))
    return videos


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    vids = build()
    out.write_text(json.dumps(vids, indent=1), encoding="utf-8")
    print(f"wrote {len(vids)} synthetic videos from {len(EDUCATORS)} educators -> {out}")
