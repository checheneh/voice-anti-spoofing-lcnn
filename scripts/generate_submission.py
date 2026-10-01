import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import ASVspoofDataset
from src.models.lcnn import LCNN


def generate_submission(model, eval_loader, device, output_file="submission.csv"):
    model.eval()
    results = []
    print("Генерация предсказаний для Eval")

    with torch.no_grad():
        for spectrogram, _, filenames in tqdm(eval_loader, desc="Eval"):
            spectrogram = spectrogram.to(device)
            logits = model(spectrogram)
            scores = (logits[:, 0] - logits[:, 1]).cpu().numpy()

            for fname, score in zip(filenames, scores):
                fname_id = fname.replace(".flac", "")
                results.append({"filename": fname_id, "score": score})

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, header=False)
    print(f"Файл сохранен в {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--eval_protocol", required=True)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LCNN(num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = ASVspoofDataset(
        args.eval_protocol, args.eval_dir, max_frames=600, is_eval=True
    )
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )

    generate_submission(model, loader, device, output_file=args.output)