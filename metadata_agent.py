import datetime

def get_video_metadata(video_path):
    # Dummy AI-generated content
    current_time = datetime.datetime.utcnow()
    scheduled_time = (current_time + datetime.timedelta(minutes=1)).isoformat("T") + "Z"

    return {
        "title": "🎥 AI Video: Explore Automation with Python!",
        "description": "Uploaded by an AI agent using your account. Explore how AI can automate YouTube uploads.",
        "tags": ["AI", "YouTube", "Automation", "Python"],
        "privacy": "private",
        "scheduled_time": scheduled_time
    }
