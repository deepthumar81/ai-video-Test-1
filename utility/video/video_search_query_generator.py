import os
import json
import re
from datetime import datetime
from openai import OpenAI
from utility.utils import log_response, LOG_TYPE_GPT

# Initialize OpenRouter API
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OpenRouter API Key. Set OPENROUTER_API_KEY in environment variables.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    model = "mistralai/mistral-small-3.1-24b-instruct:free"
    api_key=OPENROUTER_API_KEY,
)

# Logging Directory
log_directory = ".logs/gpt_logs"

# OpenRouter Model



prompt = """# Instructions

Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms. If a caption is vague or general, consider the next timed caption for more context. If a keyword is a single word, try to return a two-word keyword that is visually concrete. If a time frame contains two or more important pieces of information, divide it into shorter time frames with one keyword each. Ensure that the time periods are strictly consecutive and cover the entire length of the video. Each keyword should cover between 2-4 seconds. The output should be in JSON format, like this: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. Please handle all edge cases, such as overlapping time segments, vague or general captions, and single-word keywords.

For example, if the caption is 'The cheetah is the fastest land animal, capable of running at speeds up to 75 mph', the keywords should include 'cheetah running', 'fastest animal', and '75 mph'. Similarly, for 'The Great Wall of China is one of the most iconic landmarks in the world', the keywords should be 'Great Wall of China', 'iconic landmark', and 'China landmark'.

Important Guidelines:

- Use only English in your text queries.
- Each search string must depict something visual.
- The depictions have to be extremely visually concrete, like 'rainy street' or 'cat sleeping'.
- 'emotional moment' ❌ BAD (not visually concrete).
- 'crying child' ✅ GOOD (visually concrete).
- The list must always contain the most relevant and appropriate query searches.
- ['Car', 'Car driving', 'Car racing', 'Car parked'] ❌ BAD (too many strings).
- ['Fast car'] ✅ GOOD (concise, relevant).
- ['Un chien', 'une voiture rapide', 'une maison rouge'] ❌ BAD (must be in English).

Note: Your response should be the response only and no extra text or data.
"""

def fix_json(json_str):
    """ Fix common JSON formatting issues. """
    json_str = json_str.replace("’", "'")  # Replace typographical apostrophes
    json_str = json_str.replace("“", "\"").replace("”", "\"").replace("‘", "\"").replace("’", "\"")  # Fix quotes
    json_str = json_str.replace('"you didn"t"', '"you didn\'t"')  # Fix escaping
    return json_str

def getVideoSearchQueriesTimed(script, captions_timed):
    """ Extracts visual search queries from video captions. """
    end = captions_timed[-1][0][1]
    try:
        out = [[[0, 0], ""]]
        while out[-1][0][1] != end:
            content = call_OpenAI(script, captions_timed).replace("'", '"')
            try:
                out = json.loads(content)
            except Exception as e:
                print("content: \n", content, "\n\n")
                print(e)
                content = fix_json(content.replace("```json", "").replace("```", ""))
                out = json.loads(content)
        return out
    except Exception as e:
        print("error in response", e)
   
    return None

def call_OpenAI(script, captions_timed):
    """ Calls OpenRouter API to generate search queries. """
    user_content = f"Script: {script}\nTimed Captions: {''.join(map(str, captions_timed))}"
    print("Content", user_content)

    response = client.chat.completions.create(
        model="mistralai/mistral-small-3.1-24b-instruct:free",
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional: For OpenRouter rankings.
            "X-Title": "<YOUR_SITE_NAME>",  # Optional: For OpenRouter rankings.
        },
        extra_body={},
        temperature=1,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content}
        ]
    )
    
    text = response.choices[0].message.content.strip()
    text = re.sub('\s+', ' ', text)
    print("Text", text)
    log_response(LOG_TYPE_GPT, script, text)
    return text

def merge_empty_intervals(segments):
    """ Merges consecutive empty time intervals with previous valid segments. """
    merged = []
    i = 0
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1
            
            if i > 0:
                prev_interval, prev_url = merged[-1]
                if prev_url is not None and prev_interval[1] == interval[0]:
                    merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
                else:
                    merged.append([interval, prev_url])
            else:
                merged.append([interval, None])
            
            i = j
        else:
            merged.append([interval, url])
            i += 1
    
    return merged
