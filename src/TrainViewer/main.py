import sys
import os

# Add project root to path so `src.*` imports work when run as a script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.TrainViewer.model import TrainModel
from src.TrainViewer.view import TrainView
from src.TrainViewer.presenter import TrainPresenter


def main():
    model = TrainModel()
    view = TrainView()
    presenter = TrainPresenter(model, view)

    view.mainloop()


if __name__ == "__main__":
    main()
