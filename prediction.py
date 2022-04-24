# -*- coding: utf-8 -*-


import os
import pickle
import pandas as pd
import sklearn


#doc_new = ['obama is running for president in 2016']

#new_text = input("Please enter the news text you want to verify: ")
#print("You entered: " + str(new_text))


#function to run for prediction
def detecting_fake_news(new_text):
#retrieving the best model for prediction call
    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([new_text])
    prob = load_model.predict_proba([new_text])

    return "The given statement is \n"+str(prediction[0])+"\n The truth probability score is: \n "+str(prob[0][1])


if __name__ == '__main__':
    detecting_fake_news(new_text)