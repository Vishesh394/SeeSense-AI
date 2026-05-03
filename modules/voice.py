import threading

import pyttsx3

_engine = None
_lock = threading.Lock()
_speech_available = True
_reported_error = False


def _get_engine():
    global _engine

    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
        _engine.setProperty("volume", 1.0)

    return _engine


def speak(text):
    global _speech_available, _reported_error

    text = str(text).strip()
    if not text or not _speech_available:
        return

    try:
        with _lock:
            engine = _get_engine()
            engine.say(text)
            engine.runAndWait()
    except Exception as exc:
        _speech_available = False
        if not _reported_error:
            print(f"[voice] Speech is unavailable on this machine/session: {exc}")
            print("[voice] Detection will continue, but spoken output is disabled.")
            _reported_error = True
