import os
import json
import csv
from datetime import datetime
import time

class DGELogger:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.logs_dir = os.path.join(model_dir, 'logs')
        # Ensure dir exists immediately
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.training_log_path = os.path.join(self.logs_dir, 'training.csv')
        self.events_log_path = os.path.join(self.logs_dir, 'events.csv')
        
        # Init Events Log if not exists
        if not os.path.exists(self.events_log_path):
            with open(self.events_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Step', 'Event', 'Details'])

        # Init Training Log if not exists
        if not os.path.exists(self.training_log_path):
            with open(self.training_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Step', 
                    'Task',
                    'Loss', 
                    'Perplexity', 
                    'Memory_MB', 
                    'Frozen_Grad_Norm', 
                    'Active_Grad_Norm'
                ])

    def log_event(self, event_type, details=None, step=0):
        """
        Logs a high-level lifecycle event to events.csv.
        """
        # Ensure dir exists before writing (redundant safety)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        details_str = json.dumps(details or {})
        
        with open(self.events_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, step, event_type, details_str])
            
    def get_training_log_path(self):
        """
        Returns the path to the single unified training log.
        """
        return self.training_log_path

    def log_training_step(self, step, task_name, loss, perplexity, memory_mb, metrics):
        """
        Appends a row to the unified training log.
        """
        frozen_norm = metrics.get('frozen_grad_norm', 0.0)
        active_norm = metrics.get('active_grad_norm', 0.0)
        
        os.makedirs(self.logs_dir, exist_ok=True)
        
        with open(self.training_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, 
                task_name,
                f"{loss:.6f}", 
                f"{perplexity:.6f}", 
                f"{memory_mb:.2f}",
                f"{frozen_norm:.9f}", 
                f"{active_norm:.6f}"
            ])
