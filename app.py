import sys
import logging
from PySide6.QtWidgets import QApplication
from app.gui.main_window import MainWindow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    app = QApplication(sys.argv)
    
    # Custom font could be added here
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
