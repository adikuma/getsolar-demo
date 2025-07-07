import os
import uuid
import wave
import tempfile
import threading
import importlib.util
from pathlib import Path
from dotenv import load_dotenv
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

ai_chatbot_path = Path(__file__).parent / "ai-chatbot.py"
spec = importlib.util.spec_from_file_location("ai_chatbot", str(ai_chatbot_path))
ai_chatbot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_chatbot)

build_workflow = ai_chatbot.build_workflow
validate_system_health = ai_chatbot.validate_system_health

# openai audio client (for stt & tts)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT = "user_input.wav"

def record_audio(output_path: str) -> None:
    pa = pyaudio.PyAudio()

    # wait for user to start recording
    input("press enter to start recording")

    # open audio stream
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("recording... press enter to stop")

    # event to signal when user stops recording
    stop_event = threading.Event()
    def wait_for_enter():
        input()
        stop_event.set()

    stopper = threading.Thread(target=wait_for_enter)
    stopper.start()

    frames = []
    # read until stop event is set
    while not stop_event.is_set():
        frames.append(stream.read(CHUNK))

    # cleanup audio stream
    stream.stop_stream()
    stream.close()
    pa.terminate()

    # write frames to wav file
    wf = wave.open(output_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    print(f"saved recording to {output_path}")

def transcribe_audio(input_path: str) -> str:
    with open(input_path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
    print(f"transcription: {resp.text}")
    return resp.text

def synthesize_and_play(text: str, instructions: str = "") -> None:
    # create temp mp3 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        mp3_path = tmp.name

    # generate tts audio
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(mp3_path)

    print("playing back ai response")
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    play(audio)
    os.remove(mp3_path)

def main():
    # validate system health
    healthy = validate_system_health()
    if not healthy:
        print("warning: some components not configured; functionality may be limited.")

    # build the langgraph workflow from ai-chatbot.py
    print("initializing chat graph…")
    graph = build_workflow()
    config = {"configurable": {"thread_id": f"voice-{uuid.uuid4()}"}}

    print("voice agent ready. say 'quit' or 'exit' to stop.\n")
    while True:
        # prompt user to either record or quit
        cmd = input("press ENTER to record (or type 'quit'): ").strip().lower()
        if cmd in ("quit", "exit"):
            print("goodbye!")
            break

        # record user speech
        record_audio(WAVE_OUTPUT)

        # transcribe to text
        user_text = transcribe_audio(WAVE_OUTPUT)
        if user_text.strip().lower() in ("quit", "exit"):
            print("goodbye!")
            break

        # run through chat graph (executes tool calls if any)
        out_state = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config
        )
        messages = out_state.get("messages", [])

        # get the assistant's ai reply
        ai_messages = [
            m for m in messages
            if getattr(m, "type", None) == "ai"
        ]
        if not ai_messages:
            continue
        reply = ai_messages[-1].content
        print(f"sunny (voice): {reply}")

        # synthesize and play the ai reply
        synthesize_and_play(
            text=reply,
            instructions="respond in sunny’s friendly, enthusiastic tone."
        )

if __name__ == "__main__":
    main()