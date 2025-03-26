import pyaudio
import wave

# Aufnahmeparameter
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 60
OUTPUT_FILENAME = "loud music.wav"

def main():
    # PyAudio-Instanz erzeugen
    audio = pyaudio.PyAudio()

    # Aufnahme-Stream öffnen
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print(f"Starte Aufnahme... (Dauer {RECORD_SECONDS} Sek.)")

    frames = []

    # Anzahl der Schleifendurchläufe, um 1 Sekunde (RECORD_SECONDS) zu erreichen
    num_frames = int(RATE / CHUNK * RECORD_SECONDS)

    for _ in range(num_frames):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Aufnahme beendet.")

    # Stream stoppen und schließen
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # WAV-Datei schreiben
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio gespeichert als: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()
