from openai import OpenAI
import httpx
from pydantic import BaseModel
import os
from tqdm import tqdm
import json

file_path = "/PATH/TO/JSON/FILES/OF/EXTRACTED/REASONING/PATH"
output_path = "/PATH/TO/SAVE/RELATIONS"

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key="YOUR_API_KEY"
)

class CalendarEvent(BaseModel):
    relations: list[str]


def get_response(msg, max_tokens = 512, stops=None):
    chat_completion = client.beta.chat.completions.parse(
        messages=msg,
        model="gpt-4o-mini-2024-07-18",
        stop=stops,
        max_tokens=max_tokens,
        temperature=0,
        response_format=CalendarEvent
    )
    return chat_completion.choices[0].message.parsed
    



data = json.load(open(file_path, "r"))
relations = []
for d in tqdm(data):
    response = get_response([
        {"role": "system", "content": f"Extract the relation from the knowledge triples (subject, relation, object) involved in each sentence, and return a list of relations that is equal in length to the given list of sentences. You only need to provide a JSON structure that contains one key relations, whose value is the list of relations."},
        {"role": "user", "content": str(d)}])
    relations.append(response.relations)
json.dump(relations, open(output_path, "w+"), indent=4)