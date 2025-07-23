# ML-pro-1
This project demonstrates how to build a basic ML-powered chatbot using Python. It uses Natural Language Processing (NLP) techniques and a machine learning model to understand user queries and respond intelligently.

🚀 Features
Intent classification using Machine Learning

Preprocessed conversational dataset (JSON format)

Predictive model trained using scikit-learn or TensorFlow

Customizable intents and responses

Easy to train and extend

🛠️ Tech Stack
Python 🐍

scikit-learn or TensorFlow / Keras
NLTK / spaCy (for text preprocessing)
Flask (for web app interface – optional)

######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
CODE

import torch

from transformers import AutoTokenizer, pipeline
model = "tiiuae/falcon-7b-instruct"
tokenizer= AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype = torch.bfloat16,
    device_map = "auto",
    trust_remote_code=True
)
prompt = "What is relativity?"
newline_token = tokenizer.encode("\n")[0]

my_name = "Alice"
your_name = "Bob"
dialog = []

while True:
    user_input = input("> ")
    dialog.append(f"{my_name}: {user_input}")
    prompt = "\n".join(dialog) + f"\n{your_name}: "
    sequences = pipeline(
        prompt,
        max_length=500,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        return_full_text=False,
        eos_token_id=newline_token,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(sequences[0]['generated_text'])
    dialog.append("Bob: "+sequences[0]['generated_text'])
    #################################################################################################################################################################
    (LINK FOR GOOGLE COLAB)
    https://colab.research.google.com/drive/1YTVSzO_45r95_5Nu-qI1E7UHGMQbo6Kl?usp=sharing#scrollTo=TMKQSlLEpe82
