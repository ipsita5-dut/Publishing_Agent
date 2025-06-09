from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import datetime

# Step 1: Define YouTube API scopes
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly"
]

# Step 2: Simulate AI agent that generates metadata for the video
def get_video_metadata(video_path: str):
    current_time = datetime.datetime.utcnow()
    scheduled_time = (current_time + datetime.timedelta(minutes=1)).isoformat("T") + "Z"  

    title = "🎬 AI-Generated: How to Automate YouTube with Python"
    description = (
        "This video was uploaded by an AI Publishing Agent.\n"
        "It automatically generated this title, description, and scheduled the video!"
    )
    tags = ["AI", "Python", "YouTube API", "Automation", "Agent"]
    privacy = "private"  

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "privacy": privacy,
        "scheduled_time": scheduled_time
    }

# Step 3: Upload video to YouTube with AI-generated metadata
def upload_video_ai():
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build("youtube", "v3", credentials=creds)

    video_path = "sample-video.mp4"
    metadata = get_video_metadata(video_path)

    # AI Decision Display
    print("\n🤖 AI-Powered Metadata Decision:")
    print("Title: ", metadata["title"])
    print("Description: ", metadata["description"])
    print("Tags: ", metadata["tags"])
    print("Privacy: ", metadata["privacy"])
    print("Scheduled Time (UTC): ", metadata["scheduled_time"])
    print("\n📦 Uploading video to YouTube...")

    body = {
        "snippet": {
            "title": metadata["title"],
            "description": metadata["description"],
            "tags": metadata["tags"],
            "categoryId": "22"  # 'People & Blogs'
        },
        "status": {
            "privacyStatus": metadata["privacy"],
            "publishAt": metadata["scheduled_time"] if metadata["privacy"] == "private" else None,
            "selfDeclaredMadeForKids": False,
        }
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)

    try:
        request = service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        response = request.execute()
        print("\n✅ Video uploaded successfully!")
        print("📺 Video ID:", response["id"])

        # Simulated Dashboard
        print("\n📊 Scheduled Video Dashboard:")
        print("="*60)
        print(f"{'Title':<30} | {'Scheduled Time':<20} | {'Status'}")
        print("-"*60)
        print(f"{metadata['title'][:28]:<30} | {metadata['scheduled_time']:<20} | ✅ Scheduled")

    except Exception as e:
        print("❌ Error during upload:", e)

# Entry point
if __name__ == "__main__":
    upload_video_ai()
