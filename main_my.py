from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv 

load_dotenv()

app = FastAPI()

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Replace 'YOUR_GOOGLE_MAPS_API_KEY' with your actual Google Maps API key
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

class TravelPreferences(BaseModel):
    destination: str
    preferences: List[str]

@app.post("/plan_trip/")
async def plan_trip(travel_prefs: TravelPreferences):
    destination = travel_prefs.destination
    preferences = travel_prefs.preferences

    # Get places of interest based on destination and preferences
    places = get_places(destination, preferences)

    if not places:
        raise HTTPException(status_code=404, detail="No places found for the given destination and preferences")

    # Generate itinerary using sentence transformer model
    itinerary = generate_itinerary(destination, places, preferences)

    return {"destination": destination, "itinerary": itinerary}

def get_places(destination: str, preferences: List[str]) -> List[Dict]:
    places = []
    for preference in preferences:
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={preference}+in+{destination}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(url)
        results = response.json().get("results", [])
        for result in results:
            places.append({
                "name": result["name"],
                "address": result["formatted_address"],
                "rating": result.get("rating", "No rating"),
                "user_ratings_total": result.get("user_ratings_total", "No ratings"),
            })
    return places

def generate_itinerary(destination: str, places: List[Dict], preferences: List[str]) -> List[Dict]:
    itinerary = []
    preference_embeddings = model.encode(preferences, convert_to_tensor=True)

    for place in places:
        place_name = place["name"]
        place_description = f"{place['name']} located at {place['address']} with a rating of {place['rating']} based on {place['user_ratings_total']} reviews."
        place_embedding = model.encode(place_description, convert_to_tensor=True)

        # Calculate similarity between place description and user preferences
        similarity = util.pytorch_cos_sim(place_embedding, preference_embeddings).mean().item()

        itinerary.append({
            "place": place_name,
            "address": place["address"],
            "rating": place["rating"],
            "user_ratings_total": place["user_ratings_total"],
            "similarity": similarity
        })

    # Sort the itinerary by similarity score
    itinerary.sort(key=lambda x: x["similarity"], reverse=True)

    return itinerary

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
