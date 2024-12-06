#importing librareis
import os
import random
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
from fastapi import Request


# creating api object
app = FastAPI()
# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Load image metadata from JSON
with open("app/questions.json", "r") as f:
    image_metadata = json.load(f)
PATCH_SIZE = 256  # Grid size for shuffling
original_positions = None  # Track original positions globally
shuffled_positions = None  # Track shuffled positions globally
patches = None  # Store the patches for reshuffling
current_level = "level_1"  # Default starting level
current_image_name = None  # Global variable to track the current image

#defein the root url
@app.get("/")
def serve_html():
    """Serve the main HTML file."""
    return FileResponse("app/static/index.html")



@app.get("/image")
def get_image():
    """Serve a random image from the current level and its metadata."""
    global current_level, current_image_name

    if current_level not in image_metadata:
        return JSONResponse({"error": "Level data not found"}, status_code=404)

    level_data = image_metadata[current_level]
    if not level_data:
        return JSONResponse({"error": "No images available in the current level"}, status_code=404)

    # Select a random image
    image_name = random.choice(list(level_data.keys()))
    current_image_name = image_name  # Save selected image name

    # Build file path
    image_path = f"app/static/images/{current_level}/{image_name}"
    print(f"Attempting to load image from path: {image_path}")  # Debugging output

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")  # Debugging output
        return JSONResponse({"error": f"Image not found: {image_path}"}, status_code=404)

    # Return image metadata
    metadata = level_data[image_name]
    return JSONResponse({"image_url": f"/static/images/{current_level}/{image_name}", "metadata": metadata})




##########
from PIL import ImageOps
@app.post("/shuffle")
def shuffle_image():
    """Shuffle the image into a grid."""
    global original_positions, shuffled_positions, patches, current_image_name

    # Set patch size dynamically: 2x2 for Level 1, 4x4 for all others
    if current_level == "level_1":
        patch_size = 512 // 4  # grid
    else:
        patch_size = 512 // 4  # 4x4 grid

    # Debugging output
    print(f"Level: {current_level}, PATCH_SIZE: {patch_size}")

    # Ensure a valid image is selected
    if current_image_name is None:
        return JSONResponse({"error": "No image selected"}, status_code=400)

    image_path = f"app/static/images/{current_level}/{current_image_name}"

    # Check if the selected image exists
    if not os.path.exists(image_path):
        return JSONResponse({"error": "Image not found"}, status_code=404)

    # Open and process the image
    img = Image.open(image_path).resize((512, 512)).convert("L")  # Grayscale conversion
    img_array = np.array(img)

    # Divide the image into patches
    h, w = img_array.shape[:2]
    patches = (
        img_array.reshape(h // patch_size, patch_size, -1, patch_size)
        .swapaxes(1, 2)
        .reshape(-1, patch_size, patch_size)
    )
    original_positions = list(range(len(patches)))  # Original positions
    shuffled_positions = original_positions.copy()

    # Shuffle the positions
    random.shuffle(shuffled_positions)

    # Reconstruct the shuffled image
    shuffled_image = np.zeros_like(img_array)
    index = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            shuffled_image[i:i + patch_size, j:j + patch_size] = patches[shuffled_positions[index]]
            index += 1

    # Save the shuffled image
    shuffled_image_path = "app/static/images/shuffled_image.jpg"
    Image.fromarray(shuffled_image).save(shuffled_image_path)

    return JSONResponse({"shuffled_image_url": "/static/images/shuffled_image.jpg"})



##########



#######################
@app.post("/swap")
def swap_patches(index1: int = Form(...), index2: int = Form(...)):
    """
    Swap two patches in the shuffled image and return the updated image URL.
    """
    global shuffled_positions, patches, current_image_name, current_level

    # Ensure valid data exists for swapping
    if shuffled_positions is None or patches is None:
        return JSONResponse({"error": "No puzzle to swap"}, status_code=400)

    # Set patch size dynamically: 2x2 for Level 1, 4x4 for all others
    if current_level == "level_1":
        patch_size = 512 // 4  # 2x2 grid
    else:
        patch_size = 512 // 4  # 4x4 grid

    # Validate indices
    total_patches = len(shuffled_positions)
    if index1 < 0 or index2 < 0 or index1 >= total_patches or index2 >= total_patches:
        return JSONResponse({"error": "Invalid indices"}, status_code=400)

    # Debugging output
    print(f"Before swap: {shuffled_positions}")
    print(f"Swapping patches: {index1} <-> {index2}")

    # Swap positions in the shuffled list
    shuffled_positions[index1], shuffled_positions[index2] = shuffled_positions[index2], shuffled_positions[index1]

    # Debugging output
    print(f"After swap: {shuffled_positions}")

    # Reconstruct the updated image based on the new shuffled positions
    img_size = 512  # Assuming all images are resized to 512x512
    updated_image = np.zeros((img_size, img_size), dtype=np.uint8)  # Grayscale image

    index = 0
    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            patch = patches[shuffled_positions[index]]

            # Ensure patch size matches the calculated patch size
            if patch.shape != (patch_size, patch_size):
                raise ValueError(
                    f"Patch at index {index} has invalid dimensions: {patch.shape}. Expected: ({patch_size}, {patch_size})"
                )

            updated_image[i:i + patch_size, j:j + patch_size] = patch
            index += 1

    # Save the updated image
    updated_image_path = "app/static/images/updated_image.jpg"
    Image.fromarray(updated_image).save(updated_image_path)

    return JSONResponse({"updated_image_url": "/static/images/updated_image.jpg"})

#####################



@app.post("/validate")
def validate_puzzle():
    """
    Validate the current arrangement of the grid.
    Checks if the shuffled_positions match the original_positions.
    """
    global shuffled_positions, original_positions

    # Ensure the positions are initialized
    if original_positions is None or shuffled_positions is None:
        return JSONResponse({"error": "No puzzle to validate"}, status_code=400)

    # Check if the shuffled positions match the original positions
    is_correct = shuffled_positions == original_positions

    # Return the validation result
    return JSONResponse({"is_correct": is_correct})

@app.get("/questions")
def get_questions():
    """Serve questions related to the current image."""
    global current_level, current_image_name

    if current_image_name is None or current_level not in image_metadata:
        return JSONResponse({"error": "No image or level selected"}, status_code=400)

    # Fetch questions for the current image
    level_data = image_metadata[current_level]
    image_data = level_data.get(current_image_name)
    if not image_data:
        return JSONResponse({"error": "Image data not found"}, status_code=404)

    # Return the questions
    return {"questions": image_data.get("questions", [])}


# Check answeres
@app.post("/check_answers")
async def check_answers(request: Request):
    """Validate the player's answers."""
    global current_image_name

    # Log the incoming request
    print("Checking answers...")

    # Ensure an image is selected
    if current_image_name is None:
        print("Error: No image selected.")
        return JSONResponse({"error": "No image selected"}, status_code=400)

    # Get the questions for the current image
    level_data = image_metadata[current_level]
    image_data = level_data.get(current_image_name)
    if not image_data:
        print("Error: Image data not found.")
        return JSONResponse({"error": "Image data not found"}, status_code=404)

    # Parse player's answers from the request body
    data = await request.json()
    print(f"Received data: {data}")
    player_answers = data.get("answers", [])

    # Validate answers
    questions = image_data["questions"]
    score = 0
    detailed_results = []

    for player_answer in player_answers:
        index = player_answer.get("index")
        answer = player_answer.get("answer")

        if index is not None and index < len(questions):
            correct_answer = questions[index]["answer"]
            is_correct = answer == correct_answer
            if is_correct:
                score += 1  # Increment score for each correct answer
            detailed_results.append({
                "question": questions[index]["question"],
                "player_answer": answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            })

    print(f"Score: {score}, Detailed Results: {detailed_results}")

    # Return the result
    return JSONResponse({"score": score, "total_questions": len(questions), "details": detailed_results})




@app.post("/next_level")
def next_level():
    """Progress to the next level."""
    global current_level, current_image_name

    levels = list(image_metadata.keys())
    current_index = levels.index(current_level) if current_level in levels else -1

    if current_index < len(levels) - 1:
        current_level = levels[current_index + 1]
        current_image_name = None  # Reset current image for the new level
        return JSONResponse({"message": "Progressed to the next level", "level": current_level})
    else:
        return JSONResponse({"message": "You have completed all levels!", "level": None})
    
    


@app.on_event("startup")
def ensure_player_data_file():
    """Ensure the player_data.json file exists at startup."""
    player_data_path = "app/player_data.json"
    if not os.path.exists(player_data_path):
        with open(player_data_path, "w") as f:
            json.dump({"players": []}, f, indent=4)
        print("player_data.json file created.")




from datetime import datetime

@app.post("/save_score")
async def save_score(request: Request):
    """Save the player's score for the current level with a timestamp."""
    data = await request.json()
    print("Received data at /save_score:", data)  # Log received data

    player_id = data.get("player_id")
    level = data.get("level")
    score = data.get("score")

    print(f"Player ID: {player_id}, Level: {level}, Score: {score}")  # Detailed log

    if not player_id or not level or score is None:
        return JSONResponse({"error": "Invalid data"}, status_code=400)

    player_data_path = "app/player_data.json"
    try:
        # Load or initialize player data
        if os.path.exists(player_data_path):
            with open(player_data_path, "r") as f:
                player_data = json.load(f)
        else:
            player_data = {"players": []}

        # Find the player or create a new one
        player = next((p for p in player_data["players"] if p["player_id"] == player_id), None)
        if not player:
            player = {"player_id": player_id, "scores": {}, "total_score": 0, "timestamps": {}}
            player_data["players"].append(player)

        # Log current scores before update
        print(f"Scores before update: {player['scores']}")

        # Update the player's score
        player["scores"][level] = score

        # Get the current timestamp and store it
        timestamp = datetime.now().isoformat()
        player["timestamps"][level] = timestamp

        # Log current scores after update
        print(f"Scores after update: {player['scores']}")
        print(f"Timestamps after update: {player['timestamps']}")

        # Update total score
        player["total_score"] = sum(player["scores"].values())

        # Save the updated data
        with open(player_data_path, "w") as f:
            json.dump(player_data, f, indent=4)

        return JSONResponse({
            "message": "Score saved",
            "total_score": player["total_score"],
            "timestamp": timestamp
        })
    except Exception as e:
        return JSONResponse({"error": f"Failed to save score: {str(e)}"}, status_code=500)






####DATABASE###
import sqlite3
from datetime import datetime

# Define the database file path
DATABASE_FILE = "app/game.db"

# Function to initialize the database
def init_db():
    """Initialize the SQLite database with required tables."""
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    # Create players table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            total_score INTEGER DEFAULT 0
        )
    """)

    # Create scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            level TEXT NOT NULL,
            score INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        )
    """)

    # Commit changes and close the connection
    connection.commit()
    connection.close()
    print("Database initialized successfully.")

# Call the database initialization function at startup
init_db()


### Registering Users

import uuid
from fastapi import Form
import re

@app.post("/register")
def register_player(username: str = Form(...)):
    """Register a new player with a username and generate a unique ID."""
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    # Validate input
    if not username or len(username) < 3:
        return {"error": "Username must be at least 3 characters long"}
      # Ensure username contains only letters
    if not re.match("^[a-zA-Z ]+$", username):
        return {"error": "Username must contain only letters (a-z, A-Z)"}

    # Check if the username already exists
    cursor.execute("SELECT * FROM players WHERE username = ?", (username,))
    existing_player = cursor.fetchone()
    if existing_player:
        return {"error": "Username already exists. Choose another name."}

    # Generate a unique player ID
    player_id = str(uuid.uuid4())

    # Insert the new player into the database
    cursor.execute("""
        INSERT INTO players (player_id, username)
        VALUES (?, ?)
    """, (player_id, username))

    # Commit changes and close the connection
    connection.commit()
    connection.close()

    return {"message": "Player registered successfully", "player_id": player_id, "username": username}





####Saving scores

from datetime import datetime
@app.post("/save_score")
async def save_score(request: Request):
    data = await request.json()
    player_id = data.get("player_id")
    level = data.get("level")
    score = data.get("score")

    if not player_id or not level or score is None:
        return JSONResponse({"error": "Invalid data"}, status_code=400)

    try:
        connection = sqlite3.connect(DATABASE_FILE)
        cursor = connection.cursor()

        # Insert the level score into the scores table
        cursor.execute("""
            INSERT INTO scores (player_id, level, score, timestamp)
            VALUES (?, ?, ?, ?)
        """, (player_id, level, score, datetime.now().isoformat()))

        # Update the total_score in the players table
        cursor.execute("""
            UPDATE players
            SET total_score = (
                SELECT SUM(score)
                FROM scores
                WHERE player_id = ?
            )
            WHERE player_id = ?
        """, (player_id, player_id))

        connection.commit()
        connection.close()

        return JSONResponse({"message": "Score saved successfully"})
    except Exception as e:
        return JSONResponse({"error": f"Failed to save score: {str(e)}"}, status_code=500)


###Retrieving the Winner
@app.get("/winner")
def get_winner():
    """Retrieve the player with the highest total score."""
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    # Find the player with the highest total score
    cursor.execute("""
        SELECT username, player_id, total_score
        FROM players
        ORDER BY total_score DESC
        LIMIT 1
    """)
    winner = cursor.fetchone()
    connection.close()

    if not winner:
        return {"message": "No players found"}

    return {"winner": {"username": winner[0], "player_id": winner[1], "total_score": winner[2]}}


#winnere selection
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import os
import json
from fastapi.responses import JSONResponse

PLAYER_DATA_PATH = "app/player_data.json"
WINNER_DATA_PATH = "app/winners.json"
# Function to calculate the winner
def select_winner():
    """Calculate and save the winner based on total scores and timestamps."""
    if not os.path.exists(PLAYER_DATA_PATH):
        print("No player data found.")
        return

    with open(PLAYER_DATA_PATH, "r") as f:
        player_data = json.load(f)

    players = player_data.get("players", [])

    if not players:
        print("No players found.")
        return

    # Find the highest total score
    max_score = max(player["total_score"] for player in players)

    # Filter players with the highest score
    candidates = [player for player in players if player["total_score"] == max_score]

    if len(candidates) > 1:
        # Calculate the last level completion time for each candidate
        for player in candidates:
            # Get the latest timestamp (based on the last level)
            timestamps = player.get("timestamps", {})
            last_level = max(timestamps, key=lambda level: timestamps[level], default=None)
            player["last_completion_time"] = timestamps[last_level] if last_level else "Not available"

        # Sort candidates by last level completion time (earliest wins)
        candidates.sort(key=lambda p: p.get("last_completion_time", "9999-12-31T23:59:59"))

    # Select the first candidate as the winner
    winner = candidates[0]

    # Ensure `last_completion_time` is included in the saved winner
    if "last_completion_time" not in winner:
        timestamps = winner.get("timestamps", {})
        last_level = max(timestamps, key=lambda level: timestamps[level], default=None)
        winner["last_completion_time"] = timestamps[last_level] if last_level else "Not available"

    # Save the winner to a file for retrieval
    with open(WINNER_DATA_PATH, "w") as f:
        json.dump({"winner": winner, "max_score": max_score}, f, indent=4)

    print(f"Winner selected: {winner}")


# Set up APScheduler to trigger winner selection after 30 minutes
# Set up APScheduler to trigger winner selection after 30 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(select_winner, "interval", minutes=30)  # Adjust timing as needed
scheduler.start()

# Endpoint to fetch the winner
@app.get("/get_winner")
def get_winner():
    """Retrieve the winner from the saved data."""
    if not os.path.exists(WINNER_DATA_PATH):
        return JSONResponse({"error": "Winner data not found."}, status_code=404)

    with open(WINNER_DATA_PATH, "r") as f:
        winner_data = json.load(f)

    # Return winner details with fallback for missing data
    return {
        "winner": {
            "player_id": winner_data["winner"].get("player_id", "Unknown"),
            "total_score": winner_data["winner"].get("total_score", 0),
            "last_completion_time": winner_data["winner"].get("last_completion_time", "Not available")
        },
        "max_score": winner_data.get("max_score", 0)
    }

# Manual winner selection endpoint
@app.post("/manual_select_winner")
def manual_select_winner():
    """Manually trigger winner selection."""
    select_winner()
    return {"message": "Winner manually selected."}


from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")  # Adjust the path if necessary

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """
    Serve the admin panel page to manage winner selection.
    """
    return templates.TemplateResponse("admin.html", {"request": request})







