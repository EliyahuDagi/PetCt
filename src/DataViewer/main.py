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
