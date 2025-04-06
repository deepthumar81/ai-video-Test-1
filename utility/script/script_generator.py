import os
from openai import OpenAI
import json

if len(os.environ.get("GROQ_API_KEY")) > 30:
    from groq import Groq
    model = "llama-3.1-8b-instant"
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        )
else:
    OPENAI_API_KEY = os.getenv('OPENAI_KEY')
    model = "gpt-4o"
    client = OpenAI(api_key=OPENAI_API_KEY)

def generate_script(topic):
    prompt = (
        """You are an expert content creator for engaging YouTube Shorts focused on delivering fascinating facts. Your goal is to produce concise (under 50 seconds, ~140 words), highly shareable, and novel fact videos.

        Format:
        - Use JSON format like this: {"script": "Your script here"}
        - Only respond with a single parsable JSON object like: {\"script\": \"...\"}. Do not include any explanation or markdown.
        
        Example:
        {"script": "Amazing facts about space!\\n- One teaspoon of a neutron star weighs six billion tons!\\n- A day on Venus is longer than a year.\\n- Space isn’t completely silent; astronauts have heard ‘space sounds’ in radio signals."}
        """
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": topic}
        ]
    )

    content = response.choices[0].message.content
    print("[DEBUG] Raw content from model:\n", content)

    try:
        # Try to load full content as JSON
        script = json.loads(content)["script"]
    except json.JSONDecodeError:
        # Try to extract the JSON part from a longer message
        json_start_index = content.find('{')
        json_end_index = content.rfind('}')
        if json_start_index == -1 or json_end_index == -1:
            raise ValueError("Could not find JSON in response:\n" + content)
        content = content[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    return script

