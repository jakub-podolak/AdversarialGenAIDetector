from logger import logger
from config import parse_args
from data import get_dataset

from torch.utils.data import DataLoader
from utils.metrics import compute_metrics
from models.roberta_large import RobertaLarge


if __name__ == "__main__":
    args = parse_args()

    # Dataset
    dataset = get_dataset(args.dataset, is_test=args.is_test)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True)

    # Model
    model = RobertaLarge(num_classes=dataset.num_classes, device=args.device).to(args.device)

    model.eval()
    for idx, data in enumerate(dataloader):
        print(data)
        # text, label = data

        # output = model(text)
        
        # logger.info(f"Batch: {idx}, Data: {data}, Output: {output}, Label: {label}")
        break

