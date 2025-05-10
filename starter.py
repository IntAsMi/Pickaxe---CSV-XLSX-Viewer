from main import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.desktopSettingsAware()
    # Set the application icon
    icon_path = resource_path("pickaxe.ico")
    app_icon = QIcon(icon_path)           
    app.setWindowIcon(app_icon)            
    viewer = Pickaxe()
    viewer.show()
    viewer.load_file() # Consider if you want to auto-load on start
    sys.exit(app.exec())