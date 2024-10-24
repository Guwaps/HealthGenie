import os
import json
import pickle
import random
import nltk
import numpy
from nltk.stem import LancasterStemmer
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

import re  # Add import for regex

# Fix for TensorFlow DistributedDataset error
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

nltk.download('punkt')

stemmer = LancasterStemmer()

# Load the intents file
with open("intents.json") as file:
    data = json.load(file)

# Check if the chatbot.pickle file exists
if os.path.exists("chatbot.pickle"):
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)
    print("Loaded data from pickle file.")
else:
    print("chatbot.pickle file not found, processing data...")
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    output_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)
        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)
    print("Created and saved chatbot.pickle.")

# Model loading or training
if os.path.exists('chatbotmodel.json'):
    with open('chatbotmodel.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    myChatModel = tf.keras.models.load_model('chatbotmodel.keras')
    myChatModel.save('chatbotmodel.keras')
    print("Loaded model from disk")
else:
    print("No model found, creating a new model...")
    myChatModel = tf.keras.Sequential()  # Corrected here
    myChatModel.add(tf.keras.layers.Dense(8, input_shape=[len(words)], activation='relu'))
    myChatModel.add(tf.keras.layers.Dense(len(labels), activation='softmax'))

    myChatModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    myChatModel.fit(training, output, epochs=1000, batch_size=8)

    model_json = myChatModel.to_json()
    with open("chatbotmodel.json", "w") as json_file:
        json_file.write(model_json)
    myChatModel.save('chatbotmodel.keras')
    print("Saved model to disk")



# Define bag_of_words and chat functions
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chatWithBot(inputText, inputTag=None):
    # Check if tag is provided; if yes, skip prediction and use the given tag
    if inputTag:
        tag = inputTag
    else:
        # Use the model to predict the tag based on the inputText
        currentText = bag_of_words(inputText, words)
        currentTextArray = [currentText]
        numpyCurrentText = numpy.array(currentTextArray)

        if numpy.all((numpyCurrentText == 0)):
            return "Sorry! I don't get that.", "unknown"

        result = myChatModel.predict(numpyCurrentText[0:1])
        result_index = numpy.argmax(result)
        tag = labels[result_index]

        # Check confidence threshold
        if result[0][result_index] <= 0.7:
            return "Sorry! I don't get that.", "unknown"

    # Debug: Print the detected tag and confidence level
    print(f"Detected tag: {tag} with confidence {result[0][result_index]}")


    for tg in data["intents"]:
        if tg["tag"] == tag:
            responses = tg["responses"]
            selected_response = random.choice(responses)
            return selected_response, tag

    return "Sorry, I couldn't understand.", "unknown"



def chat():
    print("Start talking with the chatbot (type 'quit' to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Allow users to input a tag explicitly by separating with a '|'
        # Example: "my bp is 100/80 | blood_pressure_query"
        if '|' in inp:
            inputText, inputTag = inp.split('|')
            inputText = inputText.strip()
            inputTag = inputTag.strip()
        else:
            inputText = inp
            inputTag = None

        # Check for systolic and diastolic values in the format "my bp is 100/80"
        match = re.search(r'(\d+)\s*/\s*(\d+)', inputText)
        if match:
            systolic_BP = match.group(1)  # First group (Systolic)
            diastolic_BP = match.group(2)  # Second group (Diastolic)

            # Replace the values in the user message
            inputText = inputText.replace(systolic_BP, "{Systolic_BP}").replace(diastolic_BP, "{Diastolic_BP}")

            # Call the chatbot with the modified input
            response, tag = chatWithBot(inputText, inputTag)

            # Format the bot's response to include the Systolic and Diastolic values
            response = response.replace("{Systolic_BP}", systolic_BP).replace("{Diastolic_BP}", diastolic_BP)
        else:
            # If no BP format is detected, call the chatbot
            response, tag = chatWithBot(inputText, inputTag)

        print(f"Bot (tag: {tag}): {response}")

# chat()
