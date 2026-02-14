import sys
import os
# Add project root to path to allow importing src.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model import DicomModel
from view import MainView
from presenter import Presenter

def main():
    model = DicomModel()
    view = MainView()
    presenter = Presenter(model, view)
    
    view.mainloop()

if __name__ == "__main__":
    main()
