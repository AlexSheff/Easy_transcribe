import sys
import os
import torch
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, 
                             QProgressBar, QComboBox, QFrame, QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal
from app.pipeline.transcription_pipeline import TranscriptionPipeline

class InitWorker(QThread):
    finished = Signal()
    progress = Signal(float, str)
    error = Signal(str)

    def __init__(self, pipeline, model_size=None):
        super().__init__()
        self.pipeline = pipeline
        self.model_size = model_size

    def run(self):
        try:
            if self.model_size:
                self.pipeline.change_model(
                    self.model_size,
                    progress_callback=lambda p, s: self.progress.emit(p, s)
                )
            else:
                self.pipeline.initialize(
                    progress_callback=lambda p, s: self.progress.emit(p, s)
                )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class Worker(QThread):
    finished = Signal(str)
    progress = Signal(str)
    stats_update = Signal(dict)
    error = Signal(str)

    def __init__(self, pipeline, file_paths):
        super().__init__()
        self.pipeline = pipeline
        self.file_paths = file_paths

    def run(self):
        total = len(self.file_paths)
        for i, path in enumerate(self.file_paths):
            try:
                filename = os.path.basename(path)
                self.progress.emit(f"[{i+1}/{total}] Processing: {filename}...")
                
                results, export_path = self.pipeline.process_file(path)
                
                self.stats_update.emit({
                    "last_file": filename,
                    "processed_count": i + 1,
                    "total_count": total
                })
                self.progress.emit(f"Finished: {filename} -> {os.path.basename(export_path)}")
            except Exception as e:
                self.error.emit(f"Error in {os.path.basename(path)}: {str(e)}")
        
        self.finished.emit(f"Done! Processed {total} files.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEUROMICON | Transcriber Node")
        self.resize(1200, 800)
        
        # Init state
        self.processed_count = 0
        self.total_queue = 0
        self.active_workers = [] # Keep references to prevent GC
        
        self.setup_ui()
        self.apply_styles()
        
        self.pipeline = TranscriptionPipeline()
        self.start_initialization()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(sidebar)
        
        logo = QLabel("NEUROMICON")
        logo.setObjectName("logo")
        sidebar_layout.addWidget(logo)
        
        sidebar_layout.addSpacing(30)
        
        sidebar_layout.addWidget(QLabel("WHISPER ENGINE"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        sidebar_layout.addWidget(self.model_combo)
        
        sidebar_layout.addSpacing(20)
        
        self.btn_open = QPushButton("IMPORT MEDIA")
        self.btn_open.setObjectName("action_btn")
        self.btn_open.clicked.connect(self.select_file)
        sidebar_layout.addWidget(self.btn_open)
        
        sidebar_layout.addStretch()
        
        # Hardware Info
        device_name = "NVIDIA CUDA" if torch.cuda.is_available() else "Standard CPU"
        hw_info = QFrame()
        hw_info.setObjectName("hw_card")
        hw_layout = QVBoxLayout(hw_info)
        l_hw = QLabel(f"<b>ENGINE:</b> {device_name}")
        l_hw.setStyleSheet("color: #00ffcc; font-size: 10px;")
        hw_layout.addWidget(l_hw)
        sidebar_layout.addWidget(hw_info)
        
        main_layout.addWidget(sidebar)

        # --- Main Content ---
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(20)
        
        # Dashboard Header
        dash_frame = QFrame()
        dash_frame.setObjectName("dashboard")
        dash_layout = QGridLayout(dash_frame)
        dash_layout.setSpacing(15)
        
        self.stat_total = self.create_stat_card(dash_layout, "QUEUE", "0", 0, 0)
        self.stat_processed = self.create_stat_card(dash_layout, "PROCESSED", "0", 0, 1)
        self.stat_model = self.create_stat_card(dash_layout, "ACTIVE MODEL", "BASE", 0, 2)
        
        content_layout.addWidget(dash_frame)

        # Progress / Status
        status_box = QFrame()
        status_box.setObjectName("status_box")
        status_box_layout = QVBoxLayout(status_box)
        
        self.status_label = QLabel("Initializing system...")
        self.status_label.setStyleSheet("font-size: 14px; color: #fff;")
        status_box_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_box_layout.addWidget(self.progress_bar)
        
        content_layout.addWidget(status_box)

        # Output Log
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Ready for instructions...")
        content_layout.addWidget(self.text_output)

        main_layout.addWidget(content_area)

    def create_stat_card(self, layout, label, value, r, c):
        card = QFrame()
        card.setObjectName("stat_card")
        card_layout = QVBoxLayout(card)
        l1 = QLabel(label)
        l1.setObjectName("stat_label")
        l2 = QLabel(value)
        l2.setObjectName("stat_value")
        card_layout.addWidget(l1)
        card_layout.addWidget(l2)
        layout.addWidget(card, r, c)
        return l2

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #0b0b0b; color: #e0e0e0; font-family: 'Inter', 'Segoe UI', sans-serif; }
            QFrame#sidebar { background-color: #111; border-right: 1px solid #222; padding: 25px; }
            QLabel#logo { color: #00ffcc; font-size: 24px; font-weight: 800; letter-spacing: 4px; margin-bottom: 10px; }
            
            QComboBox { 
                background: #1a1a1a; border: 1px solid #333; padding: 10px; border-radius: 8px; color: #eee;
            }
            QComboBox::drop-down { border: none; }
            
            QPushButton#action_btn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00ffcc, stop:1 #0099ff);
                color: #000; font-weight: 800; border: none; padding: 15px; border-radius: 10px; font-size: 13px;
                text-transform: uppercase;
            }
            QPushButton#action_btn:hover { background: #00ffcc; margin-top: -2px; }
            QPushButton#action_btn:disabled { background: #222; color: #444; }

            QFrame#stat_card { background-color: #141414; border: 1px solid #222; border-radius: 12px; padding: 20px; }
            QLabel#stat_label { color: #00ffcc; font-size: 9px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }
            QLabel#stat_value { color: white; font-size: 28px; font-weight: 800; }

            QFrame#status_box { background-color: #141414; border-radius: 12px; padding: 20px; border: 1px solid #222; }
            QTextEdit { 
                background-color: #0d0d0d; border: 1px solid #222; border-radius: 12px; 
                padding: 20px; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 13px; color: #aaa;
            }
            
            QProgressBar { border: none; background: #222; border-radius: 3px; height: 6px; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00ffcc, stop:1 #0099ff); border-radius: 3px; }
            
            QFrame#hw_card { background: #1a1a1a; border-radius: 8px; padding: 10px; border-left: 3px solid #00ffcc; }
        """)

    def start_initialization(self, model_size=None):
        self.btn_open.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText(f"Synchronizing {model_size or 'Core'}...")
        
        worker = InitWorker(self.pipeline, model_size)
        worker.progress.connect(self.on_init_progress)
        worker.finished.connect(self.on_init_finished)
        worker.error.connect(self.on_error)
        
        self.active_workers.append(worker)
        worker.start()

    def on_init_progress(self, val, msg):
        self.progress_bar.setValue(int(val * 100))
        self.status_label.setText(msg)

    def on_init_finished(self):
        self.btn_open.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("System Ready")
        self.stat_model.setText(self.pipeline.transcriber.model_size.upper())
        # Clean up finished worker reference
        self.active_workers = [w for w in self.active_workers if w.isRunning()]

    def on_model_changed(self, model_size):
        self.start_initialization(model_size)

    def select_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Media Files", "", "Media Files (*.mp4 *.mkv *.mp3 *.wav *.m4a)"
        )
        if file_paths:
            self.start_processing(file_paths)

    def start_processing(self, file_paths):
        self.btn_open.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.text_output.clear()
        
        self.total_queue = len(file_paths)
        self.processed_count = 0
        self.update_stats()
        
        self.text_output.append(f"<span style='color: #00ffcc;'><b>[SYSTEM]</b> Queue initialized: {len(file_paths)} files.</span>\n")
        
        worker = Worker(self.pipeline, file_paths)
        worker.progress.connect(self.on_worker_progress)
        worker.stats_update.connect(self.on_stats_update)
        worker.finished.connect(self.on_finished)
        worker.error.connect(self.on_error)
        
        self.active_workers.append(worker)
        worker.start()

    def on_worker_progress(self, msg):
        self.status_label.setText(msg)
        self.text_output.append(f"<b>[PROC]</b> {msg}")

    def on_stats_update(self, stats):
        self.processed_count = stats["processed_count"]
        self.update_stats()

    def update_stats(self):
        self.stat_total.setText(str(self.total_queue))
        self.stat_processed.setText(str(self.processed_count))

    def on_finished(self, msg):
        self.btn_open.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Batch Processing Finished")
        self.text_output.append(f"\n<span style='color: #00ffcc;'><b>[DONE]</b> {msg}</span>")
        # Clean up
        self.active_workers = [w for w in self.active_workers if w.isRunning()]

    def on_error(self, message):
        self.btn_open.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Operational Error")
        self.text_output.append(f"<span style='color: #ff5555;'><b>[ERROR]</b> {message}</span>")
        # Clean up
        self.active_workers = [w for w in self.active_workers if w.isRunning()]

    def closeEvent(self, event):
        """Ensure threads are stopped on close."""
        for worker in self.active_workers:
            worker.terminate()
            worker.wait()
        event.accept()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
