from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from metadata_agent import get_video_metadata

def get_credentials(refresh_token, client_id, client_secret, scopes):
    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri='https://oauth2.googleapis.com/token',
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes
    )
    creds.refresh(Request())
    return creds

def upload_video(video_path, refresh_token, client_id, client_secret):
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = get_credentials(refresh_token, client_id, client_secret, scopes)
    service = build("youtube", "v3", credentials=creds)

    metadata = get_video_metadata(video_path)

    body = {
        "snippet": {
            "title": metadata["title"],
            "description": metadata["description"],
            "tags": metadata["tags"],
            "categoryId": "22"
        },
        "status": {
            "privacyStatus": metadata["privacy"],
            "publishAt": metadata["scheduled_time"],
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )
    response = request.execute()
    return response
