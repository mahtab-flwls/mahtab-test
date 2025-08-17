import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from audio_separator import AudioSeparator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

class AudioSeparatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Note Speech/Singing Separator")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize separator
        self.separator = AudioSeparator()
        self.current_file = None
        self.analysis_results = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Style configuration
        self.configure_styles()
    
    def configure_styles(self):
        """Configure ttk styles for a modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('Action.TButton', 
                       background='#4CAF50', 
                       foreground='white',
                       font=('Arial', 10, 'bold'))
        style.map('Action.TButton',
                 background=[('active', '#45a049')])
        
        style.configure('Secondary.TButton',
                       background='#2196F3',
                       foreground='white',
                       font=('Arial', 10))
        style.map('Secondary.TButton',
                 background=[('active', '#1976D2')])
    
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main title
        title_label = tk.Label(self.root, 
                              text="🎵 Voice Note Speech/Singing Separator 🎤",
                              font=('Arial', 16, 'bold'),
                              bg='#f0f0f0',
                              fg='#333333')
        title_label.pack(pady=20)
        
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="Audio File Selection", padding=10)
        file_frame.pack(fill='x', padx=20, pady=10)
        
        self.file_label = tk.Label(file_frame, 
                                  text="No file selected",
                                  font=('Arial', 10),
                                  bg='white',
                                  relief='sunken',
                                  anchor='w',
                                  padx=10,
                                  pady=5)
        self.file_label.pack(fill='x', pady=(0, 10))
        
        select_button = ttk.Button(file_frame, 
                                  text="📁 Select Audio File",
                                  command=self.select_file,
                                  style='Action.TButton')
        select_button.pack(pady=5)
        
        # Analysis frame
        analysis_frame = ttk.LabelFrame(self.root, text="Analysis & Separation", padding=10)
        analysis_frame.pack(fill='x', padx=20, pady=10)
        
        button_frame = tk.Frame(analysis_frame, bg='#f0f0f0')
        button_frame.pack(fill='x')
        
        self.analyze_button = ttk.Button(button_frame,
                                        text="🔍 Analyze Audio",
                                        command=self.analyze_audio,
                                        style='Secondary.TButton')
        self.analyze_button.pack(side='left', padx=(0, 10))
        
        self.separate_button = ttk.Button(button_frame,
                                         text="✂️ Separate Speech & Singing",
                                         command=self.separate_audio,
                                         style='Action.TButton')
        self.separate_button.pack(side='left')
        
        # Progress bar
        self.progress = ttk.Progressbar(analysis_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=10)
        
        # Status label
        self.status_label = tk.Label(analysis_frame,
                                    text="Ready to process audio files",
                                    font=('Arial', 9),
                                    bg='#f0f0f0',
                                    fg='#666666')
        self.status_label.pack()
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding=10)
        self.results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Results text area with scrollbar
        text_frame = tk.Frame(self.results_frame)
        text_frame.pack(fill='both', expand=True)
        
        self.results_text = tk.Text(text_frame, 
                                   height=8, 
                                   font=('Consolas', 10),
                                   bg='white',
                                   fg='#333333',
                                   wrap='word')
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Visualization frame
        self.viz_frame = tk.Frame(self.results_frame)
        self.viz_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Initially disable buttons
        self.analyze_button.configure(state='disabled')
        self.separate_button.configure(state='disabled')
    
    def select_file(self):
        """Open file dialog to select audio file."""
        file_types = [
            ('Audio files', '*.wav *.mp3 *.flac *.m4a *.ogg'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=file_types
        )
        
        if filename:
            self.current_file = filename
            self.file_label.configure(text=os.path.basename(filename))
            self.analyze_button.configure(state='normal')
            self.separate_button.configure(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.clear_visualization()
            self.update_status("File selected. Ready for analysis.")
    
    def update_status(self, message):
        """Update status label."""
        self.status_label.configure(text=message)
        self.root.update_idletasks()
    
    def start_progress(self):
        """Start progress bar animation."""
        self.progress.start(10)
    
    def stop_progress(self):
        """Stop progress bar animation."""
        self.progress.stop()
    
    def analyze_audio(self):
        """Analyze the selected audio file in a separate thread."""
        if not self.current_file:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        
        # Disable buttons during processing
        self.analyze_button.configure(state='disabled')
        self.separate_button.configure(state='disabled')
        
        # Start analysis in separate thread
        thread = threading.Thread(target=self._analyze_audio_thread)
        thread.daemon = True
        thread.start()
    
    def _analyze_audio_thread(self):
        """Analyze audio in separate thread."""
        try:
            self.root.after(0, self.start_progress)
            self.root.after(0, self.update_status, "Analyzing audio...")
            
            # Perform analysis
            self.analysis_results = self.separator.analyze_audio(self.current_file)
            
            # Update GUI in main thread
            self.root.after(0, self._display_analysis_results)
            
        except Exception as e:
            self.root.after(0, self._handle_error, f"Analysis failed: {str(e)}")
        finally:
            self.root.after(0, self.stop_progress)
            self.root.after(0, self._re_enable_buttons)
    
    def _display_analysis_results(self):
        """Display analysis results in the GUI."""
        if not self.analysis_results:
            return
        
        results = self.analysis_results
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Display summary
        summary = f"""📊 AUDIO ANALYSIS SUMMARY
{'=' * 50}
📁 File: {os.path.basename(self.current_file)}
⏱️ Total Duration: {results['total_duration']:.2f} seconds
🗣️ Speech Duration: {results['speech_duration']:.2f} seconds ({results['speech_percentage']:.1f}%)
🎵 Singing Duration: {results['singing_duration']:.2f} seconds ({results['singing_percentage']:.1f}%)

📋 SEGMENT BREAKDOWN:
{'=' * 50}
"""
        
        self.results_text.insert(tk.END, summary)
        
        # Display segments
        for i, (start, end, label) in enumerate(results['segments']):
            start_sec = start / self.separator.sample_rate
            end_sec = end / self.separator.sample_rate
            duration = end_sec - start_sec
            
            icon = "🗣️" if label == "speech" else "🎵"
            segment_info = f"{icon} Segment {i+1}: {start_sec:.1f}s - {end_sec:.1f}s ({duration:.1f}s) - {label.upper()}\n"
            self.results_text.insert(tk.END, segment_info)
        
        # Create visualization
        self.create_visualization()
        
        self.update_status("Analysis complete. Ready for separation.")
    
    def create_visualization(self):
        """Create a visualization of the analysis results."""
        if not self.analysis_results:
            return
        
        # Clear previous visualization
        self.clear_visualization()
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
        fig.patch.set_facecolor('#f0f0f0')
        
        # Timeline visualization
        segments = self.analysis_results['segments']
        total_duration = self.analysis_results['total_duration']
        
        # Create timeline
        speech_segments = []
        singing_segments = []
        
        for start, end, label in segments:
            start_sec = start / self.separator.sample_rate
            end_sec = end / self.separator.sample_rate
            
            if label == "speech":
                speech_segments.append((start_sec, end_sec - start_sec))
            else:
                singing_segments.append((start_sec, end_sec - start_sec))
        
        # Plot segments
        for start, duration in speech_segments:
            ax1.barh(0, duration, left=start, height=0.5, color='#4CAF50', alpha=0.8, label='Speech')
        
        for start, duration in singing_segments:
            ax1.barh(0, duration, left=start, height=0.5, color='#FF9800', alpha=0.8, label='Singing')
        
        ax1.set_xlim(0, total_duration)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('Audio Segments Timeline')
        ax1.set_yticks([])
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
        # Pie chart
        speech_pct = self.analysis_results['speech_percentage']
        singing_pct = self.analysis_results['singing_percentage']
        
        if speech_pct > 0 or singing_pct > 0:
            sizes = [speech_pct, singing_pct]
            labels = ['Speech', 'Singing']
            colors = ['#4CAF50', '#FF9800']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Speech vs Singing Distribution')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.canvas = canvas  # Keep reference
    
    def clear_visualization(self):
        """Clear the visualization area."""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
    
    def separate_audio(self):
        """Separate speech and singing in a separate thread."""
        if not self.current_file:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        # Disable buttons during processing
        self.analyze_button.configure(state='disabled')
        self.separate_button.configure(state='disabled')
        
        # Start separation in separate thread
        thread = threading.Thread(target=self._separate_audio_thread, args=(output_dir,))
        thread.daemon = True
        thread.start()
    
    def _separate_audio_thread(self, output_dir):
        """Separate audio in separate thread."""
        try:
            self.root.after(0, self.start_progress)
            self.root.after(0, self.update_status, "Separating speech and singing...")
            
            # Perform separation
            speech_path, singing_path = self.separator.separate_audio(self.current_file, output_dir)
            
            # Update GUI in main thread
            self.root.after(0, self._display_separation_results, speech_path, singing_path)
            
        except Exception as e:
            self.root.after(0, self._handle_error, f"Separation failed: {str(e)}")
        finally:
            self.root.after(0, self.stop_progress)
            self.root.after(0, self._re_enable_buttons)
    
    def _display_separation_results(self, speech_path, singing_path):
        """Display separation results."""
        result_text = f"""
✅ SEPARATION COMPLETE!
{'=' * 50}
📁 Output Files:
🗣️ Speech: {speech_path}
🎵 Singing: {singing_path}

The audio has been successfully separated into speech and singing components.
"""
        
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)
        
        self.update_status("Separation complete!")
        
        messagebox.showinfo("Success", 
                           f"Audio separation complete!\n\n"
                           f"Speech saved to: {os.path.basename(speech_path)}\n"
                           f"Singing saved to: {os.path.basename(singing_path)}")
    
    def _handle_error(self, error_message):
        """Handle errors in processing."""
        self.results_text.insert(tk.END, f"\n❌ ERROR: {error_message}\n")
        self.results_text.see(tk.END)
        self.update_status("Error occurred during processing.")
        messagebox.showerror("Error", error_message)
    
    def _re_enable_buttons(self):
        """Re-enable buttons after processing."""
        if self.current_file:
            self.analyze_button.configure(state='normal')
            self.separate_button.configure(state='normal')

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = AudioSeparatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()