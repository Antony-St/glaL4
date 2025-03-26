import pyaudio
import numpy as np
import time

# Audio-Konfiguration
CHUNK = 1024         # Anzahl von Frames pro Buffer
FORMAT = pyaudio.paInt16  # 16 Bit pro Sample
CHANNELS = 1         # Mono-Eingang
RATE = 44100         # Abtastrate (Samples pro Sekunde)

def main():
    # PyAudio-Instanz
    p = pyaudio.PyAudio()

    # Stream öffnen (Eingang=USB-Mikrofon)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,          # Wir lesen von einem Input (Mikrofon)
                    frames_per_buffer=CHUNK)

    print("Aufnahme läuft... Drücke Strg+C zum Beenden.")

    try:
        while True:
            # Audiodaten aus dem Stream lesen
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # In NumPy-Array umwandeln, damit wir rechnen können
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Beispiel: RMS oder "Lautstärke" berechnen
            rms = np.sqrt(np.mean(np.square(audio_data)))
            print(f"RMS: {rms:.2f}")
            time.sleep(1)

            # Hier könntest du z.B. bestimmte Grenzwerte definieren,
            # um zu erkennen, ob gerade geklatscht wird oder nicht.
            # (Dazu ist weitere Signalverarbeitung nötig.)

    except KeyboardInterrupt:
        print("Beende Aufnahme...")

    finally:
        # Stream und PyAudio sauber schließen
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
