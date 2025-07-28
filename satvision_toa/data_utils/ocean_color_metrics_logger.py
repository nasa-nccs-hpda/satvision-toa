import csv
import os

class MetricsLogger:
    def __init__(self, base_filename):
        self.epoch_file = f"{base_filename}_epoch_metrics.csv"
        self.individual_file = f"{base_filename}_individual_metrics.csv"
        self.epoch_writer = None
        self.individual_writer = None
        self.epoch_file_handle = None
        self.individual_file_handle = None
        self.individual_headers_written = False
        self.epoch_headers_written = False
    
    def log_epoch_metrics(self, epoch_metrics_dict, individual_metrics_list, epoch):
        """Log metrics for a single epoch"""
        
        # Prepare epoch data
        epoch_row = {'epoch': epoch, **epoch_metrics_dict}
        
        # Handle epoch metrics file
        if not self.epoch_headers_written:
            self._initialize_epoch_writer(epoch_row)
            self.epoch_headers_written = True
        
        self.epoch_writer.writerow(epoch_row)
        self.epoch_file_handle.flush()
        
        # Handle individual metrics (list of dicts)
        if individual_metrics_list:
            # Add epoch to each individual metric dict
            individual_rows = []
            for metric_dict in individual_metrics_list:
                individual_row = {'epoch': epoch, **metric_dict}
                individual_rows.append(individual_row)
            
            # Initialize individual writer if first call
            if not self.individual_headers_written:
                self._initialize_individual_writer(individual_rows[0])
                self.individual_headers_written = True
            
            # Write all individual metric rows for this epoch
            for row in individual_rows:
                self.individual_writer.writerow(row)
            
            self.individual_file_handle.flush()
    
    def _initialize_epoch_writer(self, epoch_row):
        """Initialize CSV writer for epoch metrics"""
        self.epoch_file_handle = open(self.epoch_file, 'w', newline='')
        self.epoch_writer = csv.DictWriter(self.epoch_file_handle, fieldnames=epoch_row.keys())
        self.epoch_writer.writeheader()
    
    def _initialize_individual_writer(self, individual_row):
        """Initialize CSV writer for individual metrics"""
        self.individual_file_handle = open(self.individual_file, 'w', newline='')
        self.individual_writer = csv.DictWriter(self.individual_file_handle, fieldnames=individual_row.keys())
        self.individual_writer.writeheader()
    
    def close(self):
        """Close file handles"""
        if self.epoch_file_handle:
            self.epoch_file_handle.close()
        if self.individual_file_handle:
            self.individual_file_handle.close()