import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ShortClipsDataset(Dataset):
    """
    Lädt WAV-Dateien direkt als Roh-Audio (Wellenform).
    Pad/Truncate auf max_length (4410 Samples = 100ms bei 44.1 kHz).
    Ordnerstruktur:
      ./../../audios/clap
      ./../../audios/no_clap
    """

    def __init__(self, root_dir="./../../audios", max_length=4410, sample_rate=44100):
        """
        root_dir: Hauptordner mit Unterverzeichnissen 'clap' und 'no_clap'
        max_length: maximale Anzahl Samples (100ms = 4410 bei 44.1kHz)
        sample_rate: Ziel-SR, falls beim Laden resamplen nötig
        """
        self.root_dir = root_dir
        self.max_length = max_length
        self.sample_rate = sample_rate

        # Ordnernamen laut Vorgabe
        self.classes = ["clap", "no_clap"]  # clap = 0, no_clap = 1
        self.files = []
        self.labels = []

        # Alle WAV-Dateien in den zwei Klassenordnern sammeln
        for label_idx, label_name in enumerate(self.classes):
            label_folder = os.path.join(root_dir, label_name)
            for filename in os.listdir(label_folder):
                if filename.lower().endswith('.wav'):
                    self.files.append(os.path.join(label_folder, filename))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(filepath)

        # Stereo -> Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resampling (nur falls SR != 44.1k)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # waveform: (1, n_samples)
        n_samples = waveform.shape[1]

        # Normalisieren auf [-1, 1]
        max_val = waveform.abs().max().clamp_min(1e-8)
        waveform = waveform / max_val

        # Trunc/Pad auf max_length
        if n_samples > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif n_samples < self.max_length:
            pad_len = self.max_length - n_samples
            waveform = torch.cat([waveform, torch.zeros((1, pad_len))], dim=1)

        # Label + Waveform zurückgeben
        return waveform, label

# ----------------------------------------------------------
# 1D-CNN für Roh-Audio
# ----------------------------------------------------------
class AudioCNN1D(nn.Module):
    """
    1D-CNN für sehr kurze Roh-Audio-Clips (max 100ms, ~4410 Samples).
    2 Klassen: Clap(0) vs. NoClap(1).
    """
    def __init__(self, num_classes=2):
        super(AudioCNN1D, self).__init__()
        # 1D-Conv-Blöcke
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)  # 4410 -> ~1102
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)  # ~1102 -> ~275
        )
        # Ausgabe: (batch, 32, ~275) => ~ 32*275 = 8800

        self.fc1 = nn.Linear(32 * 275, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (Batch, 1, 4410)
        x = self.conv1(x)           # -> (Batch, 16, ~1102)
        x = self.conv2(x)           # -> (Batch, 32, ~275)
        x = x.view(x.size(0), -1)   # -> (Batch, 8800) [ca.]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# ----------------------------------------------------------
# Training / Validation
# ----------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)  # (Batch, 1, 4410)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_model(model, dataloader, criterion, device):
    """
    Gibt neben Loss und Accuracy auch False Positives (FP) und
    False Negatives (FN) aus.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    fp = 0  # false positives
    fn = 0  # false negatives

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * waveforms.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # FP / FN zählen (0=clap, 1=no_clap => "positiv"=clap)
            for p, l in zip(predicted, labels):
                p = p.item()
                l = l.item()
                # false positive: p=0 (clap), l=1 (no_clap)
                if p == 0 and l == 1:
                    fp += 1
                # false negative: p=1 (no_clap), l=0 (clap)
                elif p == 1 and l == 0:
                    fn += 1

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, fp, fn, total


# ----------------------------------------------------------
# Hauptablauf
# ----------------------------------------------------------
def main():
    data_root = "./../../audios"  # Angepasster Pfad
    max_length = 4410              # ~100ms bei 44.1kHz
    sample_rate = 44100
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.001

    device = torch.device("cpu") 
    print("Device:", device)

    dataset = ShortClipsDataset(
        root_dir=data_root,
        max_length=max_length,
        sample_rate=sample_rate
    )

    # 80% train, 20% val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = AudioCNN1D(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, fp, fn, total = eval_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f} | "
              f"FP: {fp}/{total}, FN: {fn}/{total}")
    
        # Abbruchkriterium: Wenn Val Acc >= 1, beende das Training
        if val_acc >= 1:
            print("Val Acc 100% erreicht. Breche Training ab...")
            break

    # Modell speichern
    torch.save(model.state_dict(), "ML_CNN_wav_faltung_netz.pth")
    print("Training abgeschlossen und Modell gespeichert.")


if __name__ == "__main__":
    main()
