from flask import Flask, redirect, request, session, url_for, render_template
import os, json
import google_auth_oauthlib.flow
from tinydb import TinyDB, Query
from upload import upload_video

app = Flask(__name__)
app.secret_key = "super_secret_key"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

db = TinyDB("db.json")
CLIENT_SECRET_FILE = "client_secrets.json"
scopes = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login")
def login():
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE, scopes=scopes
    )
    flow.redirect_uri = url_for("oauth2callback", _external=True)
    auth_url, _ = flow.authorization_url(
        access_type="offline", prompt="consent", include_granted_scopes="true"
    )
    return redirect(auth_url)

@app.route("/oauth2callback")
def oauth2callback():
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE, scopes=scopes
    )
    flow.redirect_uri = url_for("oauth2callback", _external=True)
    print("Redirect URI at runtime:", url_for("oauth2callback", _external=True))

    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials
    id_info = creds.id_token
    user_email = id_info["email"] if id_info and "email" in id_info else "unknown"

    db.upsert({
        "email": user_email,
        "refresh_token": creds.refresh_token
    }, Query().email == user_email)

    session["email"] = user_email
    return redirect(url_for("upload_page"))

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if "email" not in session:
        return redirect("/")

    if request.method == "POST":
        file = request.files["video"]
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        user = db.get(Query().email == session["email"])
        with open(CLIENT_SECRET_FILE) as f:
            secrets = json.load(f)["web"]

        # Upload using that user’s token
        response = upload_video(filepath, user["refresh_token"], secrets["client_id"], secrets["client_secret"])
        return f"<h3>✅ Uploaded!</h3><a href='https://youtube.com/watch?v={response['id']}'>Watch</a>"

    return render_template("upload.html", email=session["email"])

if __name__ == "__main__":
    app.run(debug=True)
