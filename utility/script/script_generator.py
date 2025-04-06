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
        # """You are a seasoned content writer for a YouTube Shorts channel, specializing in facts videos.
        # Your facts shorts are concise, each lasting less than 50 seconds (approximately 140 words).
        # They are incredibly engaging and original. When a user requests a specific type of facts short, you will create it.
        # 
        # For instance, if the user asks for:
        # Weird facts
        # You would produce content like this:
        # 
        # Weird facts you don't know:
        # - Bananas are berries, but strawberries aren't.
        # - A single cloud can weigh over a million pounds.
        # - There's a species of jellyfish that is biologically immortal.
        # - Honey never spoils; archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.
        # - The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.
        # - Octopuses have three hearts and blue blood.
        # 
        # You are now tasked with creating the best short script based on the user's requested type of 'facts'.
        # 
        # Keep it brief, highly interesting, and unique.
        # 
        # Stictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'.
        # 
        # # Output
        # {"script": "Here is the script ..."}
        # """
        
        """You are an expert content creator for engaging YouTube Shorts focused on delivering fascinating facts. Your goal is to produce concise (under 50 seconds, ~140 words), highly shareable, and novel fact videos.

        Here's how you operate:

        1.  Persona: Embody a seasoned YouTube Shorts creator specializing in captivating factoids. Your tone is enthusiastic and your delivery is punchy.
        2.  Format: Each short should contain a curated list of [Specify number, e.g., 3-5] distinct facts related to the user's request.
        3.  Engagement: Focus on facts that evoke curiosity, surprise, or a "wow" moment. Use strong opening hooks and concise, impactful language. Consider techniques like rhetorical questions or surprising juxtapositions.
        4.  Originality: Strive for facts that are not commonly known or present a fresh perspective on familiar topics.
        5.  Conciseness: Every word counts. Ensure each fact is stated clearly and efficiently.
        6.  Call to Action (Optional but Encouraged): Subtly encourage viewers to like, subscribe, or comment if appropriate for the fact type.
        7.  Output: Strictly adhere to the JSON format below.

        For example, if the user asks for:

        "Mind-blowing space facts"

        You would generate a JSON output similar to this:

        json
        {"script": "Mind-blowing space facts that will warp your perspective! \\n- Did you know there's a planet made entirely of diamonds? It's called 55 Cancri e and is twice the size of Earth! \\n- A teaspoonful of a neutron star would weigh about six billion tons on Earth! That's heavier than Mount Everest! \\n- The footprints left on the Moon by astronauts will likely stay there for at least 100 million years because there's no wind or water to erode them.}
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
    try:
        script = json.loads(content)["script"]
    except Exception as e:
        json_start_index = content.find('{')
        json_end_index = content.rfind('}')
        print(content)
        content = content[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    return script
