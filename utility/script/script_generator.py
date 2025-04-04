import os
import json
from openai import OpenAI

# Initialize OpenRouter API
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )

# Initialize OpenRouter API
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OpenRouter API Key. Set OPENROUTER_API_KEY in environment variables.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    model = "mistralai/mistral-small-3.1-24b-instruct:free"
    api_key=OPENROUTER_API_KEY,
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

        Stictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'.

        # Output
        {"script": "Here is the script ..."}
        """
    )

    response = client.chat.completions.create(
        model="mistralai/mistral-small-3.1-24b-instruct:free",
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
        },
        extra_body={},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": topic}
        ]
    )

    content = response.choices[0].message.content
    try:
        script = json.loads(content)["script"]
    except Exception:
        json_start_index = content.find('{')
        json_end_index = content.rfind('}')
        content = content[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    
    return script

# Example Usage
topic = "Amazing space facts"
script = generate_script(topic)
print(script)
