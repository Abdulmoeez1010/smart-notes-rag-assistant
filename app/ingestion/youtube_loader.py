from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document


def extract_video_id(url: str) -> str:
    """Extract the video ID from common YouTube URL formats."""
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError(f"Could not extract video ID from URL: {url}")


def load_transcript(url: str) -> Document:
    """
    Load a YouTube transcript as a single LangChain Document.
    Tries English first, falls back to Hindi/Urdu if unavailable.
    """
    video_id = extract_video_id(url)
    preferred_languages = ["en", "hi", "ur"]

    loader = YoutubeLoader(
        video_id=video_id,
        language=preferred_languages,
        add_video_info=False,  # keep it simple for now; can add title/author later
    )

    docs = loader.load()

    if not docs:
        raise ValueError(f"No transcript found for video: {video_id}")

    doc = docs[0]
    doc.metadata["video_id"] = video_id
    doc.metadata["source_url"] = url

    return doc


# human-readable title with every ingest
import requests

def get_video_title(video_id: str) -> str:
    """Fetch video title via YouTube's public oEmbed endpoint — no API key required."""
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        res = requests.get(url, timeout=5)
        return res.json().get("title", video_id)
    except Exception:
        return video_id  # fallback if oEmbed fails for any reason
