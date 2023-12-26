---
title: Sentiment Analysis
emoji: 😻
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
license: unknown
---

# ToxicTweet-Project
## Milestone 1

The operating system that is being used is Windows 10 Home. In order to run Docker on this operating system, a
Windows Subsystem for Linux (WSL) must be used.

WSL was installed by using the wsl --install command in the Windows Command Prompt. This installed the Ubuntu 
distribution of Linux. To set the WSL version to WSL 2, the comman wsl --set-version Ubuntu 2 was used.

The Docker app was installed from the docker website. The option for WSL 2 had to be verified in the
Docker setting. Running wsl.exe -l -v checks the versions of the distro, which we could verify that
the distro Ubuntu ran in version 2. 

We set Ubuntu as the default distro with the command wsl --set-default ubuntu

Using VSCode as the coding environment, we can enter WSL by using wsl and code in the terminal. From there
a linux command prompt can be seen, using ~ to accept new commands. Running docker run hello-world verifies
that the docker is working.

![docker](https://user-images.githubusercontent.com/33811542/227808275-baf0dec3-181c-4b04-beeb-b42c35667edb.jpg)

## Milestone 2

Hugging Face URL:
https://huggingface.co/spaces/dahongj/sentiment-analysis

Models Used:

https://huggingface.co/siebert/sentiment-roberta-large-english?text=I+like+you.+I+love+you

https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis

https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

In order to use the HuggingFace space for our application, we had to create an empty model on Huggingface 
initially. From there, we included the information box as shown on the top of this README. We created a
secret key token on github that is linked with our HuggingFace account and used that key to create an 
action file or .yml. This file ensures that everytime there is an update to main, the website on HuggingFace
would start building based on the updated code.

By using the streamlit library, we were able to incorporate the pipeline function which allows us to access
and use a pretrained model from HuggingFace with ease. We created an app interface with includes a textbox
and a selection menu which allows users to input any text into the textbox before selecting the model that
they would like to use. The models used are listed above. Each of the models output a resulting sentiment
analysis of the input text as well as a probability score, which we used with the help of the pipeline 
functionality to output back onto HuggingFace's interface for the user to see. This was done for all three
models.

## Milestone 3

Finetuned Model URL: https://huggingface.co/dahongj/finetuned_toxictweets

Hugging Face URL:
https://huggingface.co/spaces/dahongj/sentiment-analysis

Finetune python file was done on Google Colab following the documentation of HuggingFace's finetuning
process. Initially the model distilbert-base-uncased was selected. The tweet and the labels are read
into variables and ran through a Dataset class. A tokenizer for Distilbert was created.
Then using the multivariable version of the distilbert-base-uncased model because there are 6 forms 
of toxicity included in the dataset that we want to finetune for. Using the native pytorch method
of training as demonstrated on the HuggingFace documentation, the model was trained and evaluated.
Both the finetuned model and its tokenizer are saved and uploaded onto HuggingFace.

## Milestone 4

Results:
The resulting web application on HuggingFace is a sentiment analysis application that allows users
to input a text of any kind and receive results of the toxicity levels. The first three pretrained
model has only two variances in the output, stating whether the text is majority positive or negative
as well as the degree that it is so. The fourth option on the selection bar allows users to select
our finetuned model, which determines six levels of toxicity: toxic, severe_toxic, obscene, insult,
threat, and identity_hate. This option determines the initial toxicity level as well as the second
highest level of toxicity. An example of 10 texts and their results are shown as an image on the website.

The landing page for the application and a video to demonstrate how to use the application are included
on this github.

Landing Page: https://sites.google.com/nyu.edu/toxtweet/home


https://github.com/dahongj/ToxicTweet/assets/33811542/79d0bdd7-136f-4fc9-bf22-1f561d323385

