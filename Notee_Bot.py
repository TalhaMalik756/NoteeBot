import os
import sys
import tkinter as tk
import mysql.connector
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PyPDF2 import PdfFileReader, PdfReader
import io

# Database configuration
db_config = {
    'user': 'root',
    'password': 'Talha@756',
    'host': 'localhost',
    'database': 'notes',
    'port': 3306
}

# Connect to the MySQL database
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

# Train the model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add a special padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Prepare the dataset
pdf_files = [
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 01 (NormalD).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 02 (SDA).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 03 (S&M).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 04 (Veri&Vali).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 05 (DatabaseSys).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 06 (InfoSec).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 07 (SPM).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 08 (SPM).pdf',
    r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\Notes/Lecture 09 (SRE).pdf'
]

dataset = []

for pdf_file in pdf_files:
    with open(pdf_file, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            dataset.append(text)

# Tokenize the dataset
train_data = tokenizer(dataset, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs = train_data['input_ids']
labels = inputs.clone()
labels[range(len(labels)), range(len(labels))] = -100

# Convert model to training mode
model.train()

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Define batch size
batch_size = 8  # Adjust batch size as needed to ensure it evenly divides the dataset size

# Ensure dataset size is divisible by batch size
assert len(inputs) % batch_size == 0, "Dataset size not divisible by batch size"

num_epochs = 3

for epoch in range(num_epochs):
    epoch_loss = 0

    for batch_start in range(0, len(inputs), batch_size):
        batch_end = min(batch_start + batch_size, len(inputs))  # Ensure batch end is within range
        optimizer.zero_grad()
        outputs = model(inputs[batch_start:batch_end], labels=labels[batch_start:batch_end])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss / (len(inputs) / batch_size)}')

# Save the trained model
torch.save(model.state_dict(), r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\TrainedModel/notee_bot.pt')

# Load the trained model
model.load_state_dict(torch.load(r'D:\PROGRAMMING\Eziline Internship\Notee_Bot\TrainedModel/notee_bot.pt'))

# GUI side code
window = tk.Tk()
window.title('nOTEE_Bot - Personal Knowledge Assistant')

# Create a label and text entry for user input
input_label = tk.Label(window, text='Enter your question:')
input_label.pack()
input_entry = tk.Entry(window, width=50, borderwidth=3)
input_entry.pack(pady=(10, 0))

def get_response(user_input):
    # Tokenize the user's input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a sequence of tokens using the trained model
    output = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id, num_beams=5, early_stopping=True)

    # Convert the generated tokens back to human-readable text
    response = tokenizer.decode(output[0])

    return response

# Create a function to handle user input
def handle_input():
    user_input = input_entry.get()
    # Call the `get_response` function to generate a response
    response = get_response(user_input)
    # Clear the input entry
    input_entry.delete(0, 'end')
    # Display the response
    output_label.config(text=response)

# Create a label to display the bot's response
output_label = tk.Label(window, text='', wraplength=400, justify='left')
output_label.pack(pady=(10, 0))

# Create a button to submit user input
submit_button = tk.Button(window, text='Submit', command=handle_input)
submit_button.pack(pady=(10, 0))

# Run the tkinter main loop
window.mainloop()
