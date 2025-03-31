import os
import json
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set up a temporary directory for caching to force a fresh download every run.
cache_dir = tempfile.mkdtemp()

# Define the model repository name for Mistral-7B-Instruct-v0.2.
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Download the tokenizer and model directly from the server with force_download.
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    force_download=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    force_download=True
)

# Create the offline text generation pipeline.
offline_text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

def generate_script(topic):
    prompt = (
        """You are a seasoned content writer for a YouTube Shorts channel, specializing in facts videos. 
Your facts shorts are concise, each lasting less than 50 seconds (approximately 140 words). 
They are incredibly engaging and original. When a user requests a specific type of facts short, you will create it.

For instance, if the user asks for:
Weird facts
You would produce content like this:

Weird facts you don't know:
- Bananas are berries, but strawberries aren't.
- A single cloud can weigh over a million pounds.
- There's a species of jellyfish that is biologically immortal.
- Honey never spoils; archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.
- The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.
- Octopuses have three hearts and blue blood.

You are now tasked with creating the best short script based on the user's requested type of 'facts'.

Keep it brief, highly interesting, and unique.

Strictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'.

# Output
{"script": "Here is the script ..."}
"""
    )
    # Combine the prompt with the topic provided by the user.
    full_prompt = prompt + "\n" + topic

    # Generate text using the offline model.
    result = offline_text_generator(full_prompt, max_length=500, num_return_sequences=1)
    content = result[0]['generated_text']

    try:
        # Attempt to extract a JSON object from the generated text.
        json_start_index = content.find('{')
        json_end_index = content.rfind('}') + 1
        script_json = json.loads(content[json_start_index:json_end_index])
        script = script_json["script"]
    except Exception as e:
        print("Error parsing output:", e)
        # Fallback: Return the full generated text if parsing fails.
        script = content

    return script

# Example usage:
if __name__ == "__main__":
    topic = "Weird facts"
    script = generate_script(topic)
    print("Generated Script:\n", script)
