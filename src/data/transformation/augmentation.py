# JP Morgan this file is for you <333333
# Or Mina :D 

from torchvision.transforms import Compose
import random

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

class RandomWordSwap:
    def __call__(self, text):
        words = text.split()
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            # Swap the words
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

class RandomCaseChange:
    def __call__(self, text):
        words = list(text)
        idx = random.randint(0, len(words) - 1)
        words[idx] = words[idx].upper() if words[idx].islower() else words[idx].lower()
        return ''.join(words)
    
def create_augmentation_pipeline():
    """
    Create a pipeline of augmentations to apply to the data.
    """
    return Compose([
        RandomSpaces(),
        RandomTypos(),
        RandomWordDeletion(),
        RandomWordSwap(),
        RandomCaseChange(),
    ])
