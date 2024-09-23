from torch.utils.data import DataLoader

from config import parse_args
from data import get_dataset
from models.roberta_detector import classify_text


if __name__ == '__main__':
    args = parse_args()
    
    dataset = get_dataset(args.dataset, is_test=args.is_test)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        texts, labels = batch
        for text in texts:
            prediction = classify_text(text)
            print(f"Prediction: {prediction}")
