"""
voice_input.py
──────────────
Captures audio from the microphone and transcribes it to text
using the SpeechRecognition library.

Backends supported (in preference order):
  1. Google Web Speech API  (online, free, no key needed)
  2. Sphinx                 (offline fallback, needs pocketsphinx)

Install:  pip install SpeechRecognition pyaudio
"""

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


def listen_from_microphone(timeout: int = 5, phrase_time_limit: int = 15) -> str:
    """
    Listen through the default system microphone and return the transcribed text.

    Args:
        timeout          : Seconds to wait for speech to start.
        phrase_time_limit: Maximum seconds for a single phrase.

    Returns:
        Transcribed text string, or empty string on failure.
    """
    if not _SR_AVAILABLE:
        print("[Voice] SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio")
        return ""

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8   # seconds of silence = end of phrase

    with sr.Microphone() as source:
        print("[Voice] Adjusting for ambient noise …")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[Voice] Listening … (speak now)")

        try:
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            )
        except sr.WaitTimeoutError:
            return ""  # no speech detected

    # Try Google Web Speech API first
    try:
        text = recognizer.recognize_google(audio)
        print(f"[Voice] Recognised: {text}")
        return text

    except sr.UnknownValueError:
        return ""  # speech detected but unintelligible

    except sr.RequestError as exc:
        # Google API unavailable → try offline Sphinx
        print(f"[Voice] Google API error ({exc}), trying offline Sphinx …")
        try:
            return recognizer.recognize_sphinx(audio)
        except Exception:
            return ""


def is_microphone_available() -> bool:
    """Check whether a microphone device is accessible."""
    if not _SR_AVAILABLE:
        return False
    try:
        with sr.Microphone():
            return True
    except (OSError, AttributeError):
        return False
