from openai import OpenAI
import os
import json
import re
from datetime import datetime
from utility.utils import log_response,LOG_TYPE_GPT

if len(os.environ.get("GROQ_API_KEY")) > 30:
    from groq import Groq
    model = "llama-3.1-8b-instant"
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        )
else:
    model = "gpt-4o"
    OPENAI_API_KEY = os.environ.get('OPENAI_KEY')
    client = OpenAI(api_key=OPENAI_API_KEY)

log_directory = ".logs/gpt_logs"


# """# Instructions

# Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms. If a caption is vague or general, consider the next timed caption for more context. If a keyword is a single word, try to return a two-word keyword that is visually concrete. If a time frame contains two or more important pieces of information, divide it into shorter time frames with one keyword each. Ensure that the time periods are strictly consecutive and cover the entire length of the video. Each keyword should cover between 2-4 seconds. The output should be in JSON format, like this: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. Please handle all edge cases, such as overlapping time segments, vague or general captions, and single-word keywords.

# For example, if the caption is 'The cheetah is the fastest land animal, capable of running at speeds up to 75 mph', the keywords should include 'cheetah running', 'fastest animal', and '75 mph'. Similarly, for 'The Great Wall of China is one of the most iconic landmarks in the world', the keywords should be 'Great Wall of China', 'iconic landmark', and 'China landmark'.

# Important Guidelines:

# Use only English in your text queries.
# Each search string must depict something visual.
# The depictions have to be extremely visually concrete, like rainy street, or cat sleeping.
# 'emotional moment' <= BAD, because it doesn't depict something visually.
# 'crying child' <= GOOD, because it depicts something visual.
# The list must always contain the most relevant and appropriate query searches.
# ['Car', 'Car driving', 'Car racing', 'Car parked'] <= BAD, because it's 4 strings.
# ['Fast car'] <= GOOD, because it's 1 string.
# ['Un chien', 'une voiture rapide', 'une maison rouge'] <= BAD, because the text query is NOT in English.

# Note: Your response should be the response only and no extra text or data.
#   """
prompt = """ # Instructions


Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms. If a caption is vague or general, consider the next timed caption for more context. If a keyword is a single word, try to return a two-word keyword that is visually concrete. If a time frame contains two or more important pieces of information, divide it into shorter time frames with one keyword each. Ensure that the time periods are strictly consecutive and cover the entire length of the video. Each keyword should cover between 4–8 seconds. Prefer longer durations if visuals remain consistent.
The output should be in JSON format, like this: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. Please handle all edge cases, such as overlapping time segments, vague or general captions, and single-word keywords.

Here's how to approach it:

 Visual Concreteness is Key: Each keyword or keyword phrase must represent something that can be clearly visualized. Think of what you would type into an image or video search engine.
     Good Examples: rainy street, cat sleeping, cheetah running, Great Wall of China, red apple, blue ocean waves.
     Bad Examples: emotional moment, important information, interesting fact, concept, feeling.
 Specificity: Be as specific as possible within a concise phrase. Instead of animal, use cheetah. Instead of building, use Great Wall of China.
 Conciseness: Keep keywords short and to the point (ideally 1-3 words).
 Relevance: Keywords should accurately reflect the visual content described in the caption.
 Time Segmentation:
     Process the captions sequentially.
     Each keyword should ideally represent the visual content for approximately 2-4 seconds of the video. Adjust this based on the density of visual information. Shorter segments might only need one strong keyword. Longer segments with multiple visual elements should be broken down into shorter time frames, each with its own keyword(s).
     Ensure the time periods in your output are strictly consecutive and cover the entire video duration. Represent time in seconds.
 Handling Vague Captions: If a caption is too general or abstract to yield concrete visuals, use the context from the immediately preceding or succeeding captions to infer potential visual elements.
 Single Word Keywords: If a single word strongly represents a visual (e.g., cheetah), use it. You don't always need two words. However, if a single word is too broad, try to add a visually specific modifier (e.g., car becomes red car).
 Multiple Visuals in One Segment: If a time segment contains two or more distinct and important visual elements, either create multiple keywords for that segment or, if the segment is long enough (more than 8 seconds), consider splitting it into shorter time frames, each focusing on one visual.
 Language: Use only English keywords.
 Output Format: Return the keywords in JSON format as a list of lists. Each inner list contains:
     A list of two numbers representing the start and end time (in seconds) of the segment.
     A list of 1 to 2 keyword strings per segment to reduce visual clutter.

Important Guidelines:

 Use only English in your text queries.
 Each search string must depict something visual.
 The depictions have to be extremely visually concrete, like rainy street, or cat sleeping.
 'emotional moment' <= BAD
 'crying child' <= GOOD
 The list must always contain the most relevant and appropriate query searches.
 ['Car'] <= GOOD
 ['Car driving'] <= GOOD
 ['Car racing'] <= GOOD
 ['Car parked'] <= GOOD (Each is a distinct visual)
 ['Un chien', 'une voiture rapide', 'une maison rouge'] <= BAD (Not English)

Note: Your response should be the response only and no extra text or data.
"""

def fix_json(json_str):
    # Remove all control characters except whitespace
    json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    # Normalize quotes
    json_str = json_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Try to handle unescaped double quotes inside strings
    json_str = re.sub(r'(?<!\\)"(.*?)"(?![:,}\]])', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', json_str)
    return json_str.strip()


def getVideoSearchQueriesTimed(script, captions_timed):
    end = captions_timed[-1][0][1]
    try:
        out = [[[0, 0], ""]]
        while out[-1][0][1] != end:
            content = call_OpenAI(script, captions_timed)

            # Clean known markdown formatting & extra spaces
            content = content.replace("```json", "").replace("```", "").strip()

            try:
                out = json.loads(content)
            except json.JSONDecodeError as e:
                print("RAW content (before fix):\n", content, "\n")
                print("Decode error:", e)

                # Try fallback fix
                content_fixed = fix_json(content)
                try:
                    out = json.loads(content_fixed)
                except json.JSONDecodeError as e2:
                    print("Fixed content failed too:\n", content_fixed)
                    raise e2  # Still invalid, re-raise
        return out
    except Exception as e:
        print("Error in final output:", e)
        return None


def call_OpenAI(script,captions_timed):
    user_content = """Script: {}
Timed Captions:{}
""".format(script,"".join(map(str,captions_timed)))
    print("Content", user_content)
    
    response = client.chat.completions.create(
        model= model,
        temperature=1,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content}
        ]
    )
    
    text = response.choices[0].message.content.strip()
    text = re.sub('\s+', ' ', text)
    print("Text", text)
    log_response(LOG_TYPE_GPT,script,text)
    return text

# def merge_empty_intervals(segments):
#     merged = []
#     i = 0
#     while i < len(segments):
#         interval, url = segments[i]
#         if url is None:
#             # Find consecutive None intervals
#             j = i + 1
#             while j < len(segments) and segments[j][1] is None:
#                 j += 1
            
#             # Merge consecutive None intervals with the previous valid URL
#             if i > 0:
#                 prev_interval, prev_url = merged[-1]
#                 if prev_url is not None and prev_interval[1] == interval[0]:
#                     merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
#                 else:
#                     merged.append([interval, prev_url])
#             else:
#                 merged.append([interval, None])
            
#             i = j
#         else:
#             merged.append([interval, url])
#             i += 1
    def merge_similar_segments(segments):
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if set(seg[1]) == set(last[1]):  # or use a similarity metric
                last[0][1] = seg[0][1]  # Extend end time
            else:
                merged.append(seg)
    return merged
    
    return merged
