import json
from uxsim import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QPalette, QColor, QMovie, QIntValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal,Qt
import sys
import time
from IPython.display import display, Image
import io
import world
import neural_tester as wrld
import multiprocessing
def center(self):
            frameGm = self.frameGeometry()
            screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
            centerPoint = QApplication.desktop().screenGeometry(screen).center()
            frameGm.moveCenter(centerPoint)
            self.move(frameGm.topLeft())    
class SecondaryWidget(QWidget):
    def __init__(self,gif):
        super().__init__()
        center(self)
        self.setWindowTitle("Animation")
        self.setFixedSize(600,600)
        layout = QVBoxLayout()
        label=QLabel(self)
        pic1 = QMovie(gif)  
        label.setMovie(pic1)
        label.setScaledContents(True)
        pic1.start()
        layout.addWidget(label)
        self.setLayout(layout)
class SecondaryWidgetforPic(QWidget):
    def __init__(self,pic):
        
        super().__init__()
        center(self)
        self.setWindowTitle("Pic")
        layout = QVBoxLayout()
        label=QLabel(self)
        pix = QPixmap(pic)
        label.setPixmap(pix)
        layout.addWidget(label)
        self.setLayout(layout)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Uxsim Simulation: Neural Tester")
        #Styling and colors on main window
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.black)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)
        self.setFixedSize(1100, 1000)

        #Menu Creation
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.file_menu = QMenu("Menu", self)
        open_action = QAction("Check Detailed Network for no Neural Network", self)
        open2_action = QAction("Check Detailed Network for Neural Network", self)
        many_seeds = QAction("Run simulation for Many Seeds", self)
        exit_action = QAction("Exit", self)
        #Adding actions to the Menu
        self.file_menu.addAction(open_action)
        self.file_menu.addAction(open2_action)
        self.file_menu.addAction(many_seeds)
        self.file_menu.addAction(exit_action)

        #Add menu to window
        self.menu_bar.addMenu(self.file_menu)

        #Create Combo box(Options between Neural Networks)
        self.neural_algorithms=QComboBox()
        self.neural_algorithms.addItems(["Simple Deep Neural Network","Gated Recurrent Unit Neural Network"])

        open2_action.triggered.connect(lambda: self.animatedNet("out/sim_fancy.gif")) #Show fancy network for Neural Network Simulation
        open_action.triggered.connect(lambda: self.animatedNet("out/sim_no_nn_fancy.gif")) #Show fancy network for normal simulation
        many_seeds.triggered.connect(lambda: self.run_seeds("out/run_for_many_seeds.png"))#Run function for many different seeds 
        exit_action.triggered.connect(self.close)#Close app
        #Set as main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        #Layout of pictures
        self.pic_layout = QHBoxLayout()
        textlabel = QLabel("The network of our simulation")
        textlabel.setAlignment(Qt.AlignCenter)
        self.label2 = QLabel() #label for network AND gif
        self.label=QLabel() #label for gif
        self.pic = QPixmap("out/network.png").scaled(600, 600) #Picture of network
        self.label2.setPixmap(self.pic)
        self.label2.setAlignment(Qt.AlignCenter)#set network pic to middle of the window
        #Our text fields
        self.text_field = QTextEdit()
        self.text_field.setReadOnly(True)
        self.text_field2 = QTextEdit()
        self.text_field2.setReadOnly(True)

        #Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        #Used for seed text and value
        self.text_seed=QLabel("Enter Seed value here:")
        self.line_edit = QLineEdit(self)
        self.seed=QHBoxLayout()
        self.line_edit.setFixedWidth(50)
        int_validator = QIntValidator()
        self.line_edit.setValidator(int_validator)

        #Buttons for the simulation
        self.but1 = QPushButton('Start Simulation')
        self.but2 = QPushButton('Simulation with Neural Tester')
        #Layouts for buttons and the text layout
        button_layout = QHBoxLayout()
        neural_layout=QHBoxLayout()
        text_layout = QHBoxLayout()

        #Button layout
        button_layout.addWidget(self.but1) 
        button_layout.addLayout(neural_layout)
        neural_layout.addWidget(self.but2)
        neural_layout.addWidget(self.neural_algorithms)  
        #Add text fields to layout
        text_layout.addWidget(self.text_field)
        text_layout.addWidget(self.text_field2)
        #Add layouts
        main_layout.addLayout(self.pic_layout)
        main_layout.addWidget(textlabel)
        self.pic_layout.addWidget(self.label2)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(text_layout)

        #Seed value location on window
        self.seed.addStretch() 
        self.seed.addWidget(self.text_seed)
        self.seed.addWidget(self.line_edit)
        self.seed.addStretch()
        self.but1.clicked.connect(self.start_simulation_no_nn)
        self.but2.clicked.connect(self.start_simulation)
        #add last layout and progress bar
        main_layout.addLayout(self.seed)
        main_layout.addWidget(self.progress_bar)
        center(self)
    #function used to run simulation without neural network
    def run_no_nn(self):
        #grab text of Combo Box
        seed=self.line_edit.text()
        self.text_field.setText("")
        self.text_field.repaint()
        if(seed==""):
            self.text_field.setText("invalid seed")
            return 0
        sim_no_nn=wrld.init_simulation_without_nn(int(seed),"No neural")#Run simulation without NN
        self.progress_bar.setValue(0)
        stats=sim_no_nn.simple_stats_to_json()
        self.progress_bar.setValue(10)
        #Save both fancy and simple network animations named sim_no_nn and sim_no_nn_fancy
        sim_no_nn.W.analyzer.network_anim(animation_speed_inverse=20, timestep_skip=50, detailed=0, network_font_size=0, figsize=(4,4),file_name="out/sim_no_nn.gif")
        sim_no_nn.W.analyzer.network_fancy(animation_speed_inverse=15, sample_ratio=0.3, interval=3, trace_length=3, network_font_size=0,file_name="out/sim_no_nn_fancy.gif")

        self.progress_bar.setValue(40)

        key_mapping = {
            'average_speed': 'Average Speed',
            'completed_trips': 'Completed Trips',
            'total_trips': 'Total Trips',
            'delay': 'Delay',
            'delay_ratio': 'Delay Ratio',
            'average_traffic_volume': 'Average Traffic Volume'
        }
        #Print stats
        valuepr=0
        self.text_field.append(f"Process ran on seed {seed}")
        for key, label in key_mapping.items():
            value = stats[key]
            time.sleep(0.5)
            valuepr+=int(60/len(key_mapping.items()))
            self.progress_bar.setValue(40+valuepr)
            if isinstance(value, float):
                self.text_field.append(f"{label}: {float(value):.2f}")
            else:
                self.text_field.append(f"{label}: {value}")
        #End progress bar
        self.progress_bar.setValue(100)
    #Function used to run simulation with neural network
    def run_neural_tester(self):
        #Get Neural Network
        text = str(self.neural_algorithms.currentText())
        self.text_field2.setText("")
        self.text_field.repaint()
        seed=self.line_edit.text()
        if(seed==""):
            self.text_field2.setText("invalid seed")
            return 0
        self.progress_bar.setValue(0)
        sim=wrld.init_simulation_with_nn(int(seed),text) #Get simulation
        self.progress_bar.setValue(20)
        #Save both fancy and simple network animations named sim_ and sim_fancy
        sim.W.analyzer.network_anim(animation_speed_inverse=20, timestep_skip=50, detailed=0, network_font_size=0, figsize=(4,4),file_name="out/sim.gif")
        sim.W.analyzer.network_fancy(animation_speed_inverse=15, sample_ratio=0.3, interval=3, trace_length=3, network_font_size=0,file_name="out/sim_fancy.gif")
        stats=sim.simple_stats_to_json()
        self.progress_bar.setValue(40)
        key_mapping = {
            'average_speed': 'Average Speed',
            'completed_trips': 'Completed Trips',
            'total_trips': 'Total Trips',
            'delay': 'Delay',
            'delay_ratio': 'Delay Ratio',
            'average_traffic_volume': 'Average Traffic Volume'
        }
        #Print stats
        valuepr=0
        self.text_field2.append(f"Process ran on seed {seed} and using: {text}")
        for key, label in key_mapping.items():
            value = stats[key]
            time.sleep(0.5)
            valuepr+=int(60/len(key_mapping.items()))
            self.progress_bar.setValue(40+valuepr)
            if isinstance(value, float):
                self.text_field2.append(f"{label}: {float(value):.2f}")
            else:
                self.text_field2.append(f"{label}: {value}")
        self.progress_bar.setValue(100)
    #Function used to print detailed gifs on the menu
    def animatedNet(self,gif):
        self.secondary_widget = SecondaryWidget(gif)
        self.secondary_widget.show()
    #Function used to run function run_many_seeds
    def run_seeds(self,pic):
        text = str(self.neural_algorithms.currentText())
        wrld.run_for_many_seeds(text)
        self.secondary_widget = SecondaryWidgetforPic(pic)
        self.secondary_widget.show()
    #Function used to initialize Simulation and disable buttons and menu for no neural network
    def start_simulation_no_nn(self):        
        self.but1.setEnabled(False)
        self.file_menu.setEnabled(False)
        self.but2.setEnabled(False)
        self.simulation_worker = self.run_no_nn()
        if self.simulation_worker != 0:#not invalid seed
            self.reEnable_no_nn()
        else:
            self.file_menu.setEnabled(True)
            self.but1.setStyleSheet("background-color:QColor(53, 53, 53)")
            self.but1.setEnabled(True)
            self.but2.setEnabled(True)
    #Function used to initialize Simulation and disable buttons and menu for  neural network
    def start_simulation(self):        
        self.but1.setEnabled(False)
        self.file_menu.setEnabled(False)
        self.but2.setEnabled(False)
        self.simulation_worker = self.run_neural_tester()
        if self.simulation_worker != 0:#not invalid seed
            self.reEnable()
        else:
            self.file_menu.setEnabled(True)
            self.but1.setStyleSheet("background-color:QColor(53, 53, 53)")
            self.but1.setEnabled(True)
            self.but2.setEnabled(True)
    #Enable buttons and show gifs available for no neural network simulation
    def reEnable(self):
        self.pic1 = QMovie("out/sim_no_nn.gif")
        self.label2.setMovie(self.pic1)
        self.label2.setScaledContents(True)
        self.pic1.start()
        self.pic = QMovie("out/sim.gif")
        self.label.setMovie(self.pic)
        self.file_menu.setEnabled(True)
        self.pic_layout.addWidget(self.label2)
        self.pic.start()
        self.pic_layout.addWidget(self.label)
        self.but1.setStyleSheet("background-color:QColor(53, 53, 53)")
        self.but1.setEnabled(True)
        self.but2.setEnabled(True)

    #Enable buttons and show gifs available for neural network
    def reEnable_no_nn(self):
        self.pic1 = QMovie("out/sim_no_nn.gif")  
        self.label.setMovie(self.pic1)
        self.label.setScaledContents(True)
        self.pic1.start()
        self.pic = QMovie("out/sim.gif")
        self.label2.setMovie(self.pic)
        self.file_menu.setEnabled(True)
        self.pic_layout.addWidget(self.label)
        self.pic.start()
        self.pic_layout.addWidget(self.label2)
        self.but1.setStyleSheet("background-color:QColor(53, 53, 53)")
        self.but1.setEnabled(True)
        self.but2.setEnabled(True)

if __name__ == "__main__":
    #multiprocessing is spawn to solve errors of cuda processing
    multiprocessing.set_start_method('spawn')
    if not os.path.isdir("out"):
        os.mkdir("out")
    #Get network image 
    wrld.get_network("world")
    app = QApplication(sys.argv)
    window = MainWindow()
    #delete existing gifs and pics
    if os.path.exists("out/sim_no_nn.gif"):
        os.remove("out/sim_no_nn.gif")   
    if os.path.exists("out/sim_no_nn_fancy.gif"):
        os.remove("out/sim_no_nn_fancy.gif")   
    if os.path.exists("out/sim.gif"):
        os.remove("out/sim.gif") 
    if os.path.exists("out/sim_fancy.gif"):
        os.remove("out/sim_fancy.gif")   
    if os.path.exists("out/network.png"):
        os.remove("out/network.png") 
    if os.path.exists("out/run_for_many_seeds.png"):
        os.remove("out/run_for_many_seeds.png")
      
    
    #Show main window
    window.show()
    sys.exit(app.exec())










