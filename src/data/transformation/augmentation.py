# JP Morgan this file is for you <333333
# Or Mina :D 

from models.t5_paraphraser import paraphrase_text 
import random
import string
from torchvision.transforms import Compose


# random spaces
class RandomSpaces:
    def __call__(self, text):
        words = list(text)
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, ' ') 
        return ''.join(words)

# typos
class RandomTypos:
    def __call__(self, text):
        words = list(text)
        if len(words) > 1:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx] 
        return ''.join(words)

# random words deleted
class RandomWordDeletion:
    def __call__(self, text):
        words = text.split()
        if len(words) > 1:
            words.pop(random.randint(0, len(words) - 1)) 
        return ' '.join(words)

# random words swapped
class RandomWordSwap:
    def __call__(self, text):
        words = text.split()
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            # Swap the words
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

# random case change
class RandomCaseChange:
    def __call__(self, text):
        words = list(text)
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].upper() if words[idx].islower() else words[idx].lower()
        return ''.join(words)

# random spaces with probability
class RandomSpaceWithProbability:
    def __init__(self, probability: float):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.probability = probability

    def __call__(self, text: str) -> str:
        result = []
        for char in text:
            result.append(char)
            if char != ' ' and random.random() < self.probability:
                result.append(' ')
        return ''.join(result)

# random typo with probability
class RandomTypoWithProbability:
    def __init__(self, probability: float):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.probability = probability

    def __call__(self, text: str) -> str:
        result = []
        for char in text:
            if char.isalpha() and random.random() < self.probability:
                # Replace the character with a random letter to simulate a typo
                typo_char = random.choice(string.ascii_letters)
                result.append(typo_char)
            else:
                result.append(char)
        return ''.join(result)

# random neighboring word swap with probability
class RandomNeighbouringWordSwapWithProbability:
    def __init__(self, probability: float):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.probability = probability

    def __call__(self, text: str) -> str:
        words = text.split()
        i = 0
        while i < len(words) - 1:
            if random.random() < self.probability:
                # Swap two neighboring words
                words[i], words[i + 1] = words[i + 1], words[i]
                i += 2  # Skip to the next pair after the swap
            else:
                i += 1
        return ' '.join(words)

# Style Augmentations

class StyleAugmentation:
    def __init__(self):
        pass 

    def to_formal(self, text):
        return paraphrase_text(f"Paraphrase to formal: {text}")[0]  # Adjust prompt as needed

    def to_10yearold(self, text):
        return paraphrase_text(f"Paraphrase for a 10-year-old: {text}")[0]

    def to_informal(self, text):
        return paraphrase_text(f"Paraphrase to informal: {text}")[0]

    def to_scientific(self, text):
        return paraphrase_text(f"Paraphrase to scientific: {text}")[0]

    def apply_style_augmentation(self, text, style):
        if style == "to_formal":
            return self.to_formal(text)
        elif style == "to_10yearold":
            return self.to_10yearold(text)
        elif style == "to_informal":
            return self.to_informal(text)
        elif style == "to_scientific":
            return self.to_scientific(text)
        else:
            raise ValueError(f"Unknown style: {style}")


ALL_AUGMENTATIONS = [
    ("RandomSpaces", RandomSpaces()),
    ("RandomTypos", RandomTypos()),
    ("RandomWordDeletion", RandomWordDeletion()),
    ("RandomWordSwap", RandomWordSwap()),
    ("RandomCaseChange", RandomCaseChange()),
    ("StyleToFormal", StyleAugmentation().to_formal),
    ("StyleTo10YearOld", StyleAugmentation().to_10yearold),
    ("StyleToInformal", StyleAugmentation().to_informal),
    ("StyleToScientific", StyleAugmentation().to_scientific)
]

ALL_AUGMENTATIONS_PROBA = [
    ("RandomSpaceWithProbability", RandomSpaceWithProbability(0.01)),
    ("RandomTypoWithProbability", RandomTypoWithProbability(0.01)),
    ("RandomNeighbouringWordSwapWithProbability", RandomNeighbouringWordSwapWithProbability(0.02)),
]


def get_all_possible_augmentations(text: str, label: int) -> list[dict[str, str | int]]:
    """
    Creates a list of the given text transformed by all the augmentations,
    including style-based ones.
    """
    result = [{"augmentation_name": "No augmentation", "text": text, "label": label}]
    for name, augmentation in ALL_AUGMENTATIONS:
        result.append({
            "augmentation_name": name,
            "text": augmentation(text),
            "label": label
        })

    return result

    
def create_augmentation_pipeline():
    """
    Create a pipeline of augmentations to apply to the data.
    """
    return Compose([augmentation for name, augmentation in ALL_AUGMENTATIONS])


def create_augmentation_pipeline_proba():
    return Compose([augmentation for name, augmentation in ALL_AUGMENTATIONS_PROBA])

