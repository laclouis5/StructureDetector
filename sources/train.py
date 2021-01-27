from library.data import *
from library.utils import *
from library.model import *


args = Arguments().parse()
trainer = Trainer(args)
trainer.train()
