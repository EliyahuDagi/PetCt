import sys
import os

# Add project root to path so `src.*` imports work when run as a script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.TrainViewer.model import TrainModel
from src.TrainViewer.app import launch


def main():
    model = TrainModel()
    # Launch locally with the dark theme (no public share); opens the browser.
    launch(model, inbrowser=True, share=False)


if __name__ == "__main__":
    main()
