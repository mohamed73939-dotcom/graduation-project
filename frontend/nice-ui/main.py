from nicegui import ui, app
import aiohttp
import asyncio
import os

# --- Configuration ---
API_URL = "http://localhost:8000/api/summarize"
THEME_COLOR = '#5898d4' # Light Blue/Cyan for a modern tech feel
BG_Gradient = 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)'

# --- Styles ---
ui.add_head_html('''
<style>
    body {
        background: %s;
        color: #e2e8f0;
        font-family: 'Segoe UI', sans-serif;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    .gradient-text {
        background: linear-gradient(to right, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .custom-loader .q-spinner {
        color: #60a5fa !important;
    }
</style>
''' % BG_Gradient)

# --- State ---
class AppState:
    def __init__(self):
        self.processing = False
        self.progress = 0.0
        self.status_message = "Ready to upload"
        self.result = None

state = AppState()

# --- Logic ---

async def upload_and_process(file):
    state.processing = True
    state.progress = 0.1
    state.status_message = "Uploading video..."
    container_processing.refresh()
    
    try:
        data = aiohttp.FormData()
        data.add_field('video', file.file, filename=file.name, content_type='video/mp4')
        data.add_field('language', 'auto')

        state.progress = 0.3
        state.status_message = "Processing video with AI... (This may take a while)"
        container_processing.refresh()

        async with aiohttp.ClientSession() as session:
            # Huge timeout for long videos
            timeout = aiohttp.ClientTimeout(total=1200) 
            async with session.post(API_URL, data=data, timeout=timeout) as response:
                if response.status == 200:
                    json_response = await response.json()
                    state.result = json_response.get("data")
                    state.status_message = "Complete!"
                    state.progress = 1.0
                    ui.notify('Analysis Complete!', type='positive')
                    show_results()
                else:
                    error_text = await response.text()
                    ui.notify(f'Error: {response.status} - {error_text}', type='negative')
                    state.status_message = "Failed."
                    state.progress = 0.0
    except Exception as e:
        ui.notify(f'Connection Error: {str(e)}', type='negative')
        state.status_message = "Connection Failed."
        state.progress = 0.0
    finally:
        state.processing = False
        container_processing.refresh()


def handle_upload(e):
    # e.content is a standard python file-like object
    # But NiceGUI upload component handles the file separately.
    # We need to access the file from the event args. 
    # For NiceGUI 1.4+, e.content is the file object.
    asyncio.create_task(upload_and_process(e))

def show_results():
    container_upload.set_visibility(False)
    container_processing.set_visibility(False)
    container_results.set_visibility(True)
    container_results.refresh()

def reset_app():
    state.result = None
    state.processing = False
    state.progress = 0.0
    container_upload.set_visibility(True)
    container_processing.set_visibility(True) # Visible but empty/reset
    container_results.set_visibility(False)
    container_upload.refresh()
    container_processing.refresh()

# --- UI Components ---

@ui.refreshable
def container_processing():
    if not state.processing and state.progress == 0:
        return # Show nothing if not started
        
    with ui.card().classes('w-full max-w-2xl mx-auto glass-card p-6 mt-8 items-center text-center'):
        ui.label(state.status_message).classes('text-lg font-bold mb-2')
        if state.processing:
             ui.linear_progress(value=state.progress).classes('w-full h-2 rounded-full')
             ui.spinner('dots', size='lg', color='blue-400').classes('mt-4')

@ui.refreshable
def container_results():
    if not state.result:
        return

    data = state.result
    summary_html = data.get('summary_html', '')
    transcription = data.get('transcription_full', '')
    
    with ui.column().classes('w-full max-w-5xl mx-auto gap-6'):
        
        # Header for Results
        with ui.row().classes('w-full justify-between items-center'):
            ui.label('Analysis Results').classes('text-3xl font-bold gradient-text')
            ui.button('Analyze Another', on_click=reset_app, icon='refresh').props('flat color=white')

        # Tabs
        with ui.tabs().classes('w-full text-white') as tabs:
            tab_summary = ui.tab('Summary', icon='summarize')
            tab_transcript = ui.tab('Transcription', icon='description')
            # tab_slides = ui.tab('Slides', icon='image') # Disabled for now as per backend

        with ui.tab_panels(tabs, value=tab_summary).classes('w-full glass-card text-white'):
            
            # Summary Panel
            with ui.tab_panel(tab_summary):
                ui.html(summary_html).classes('prose prose-invert max-w-none text-lg leading-relaxed')
            
            # Transcription Panel
            with ui.tab_panel(tab_transcript):
                ui.textarea(value=transcription).props('readonly autogrow input-class="text-gray-300"').classes('w-full bg-transparent border-none text-base')

# --- Main Layout ---

with ui.column().classes('w-full min-h-screen items-center p-4 md:p-10 gap-8'):
    
    # Hero / Header
    with ui.column().classes('items-center text-center space-y-2'):
        ui.icon('smart_display', size='4rem', color='blue-400').classes('mb-2')
        ui.label('Sidecut AI').classes('text-5xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500')
        ui.label('Professional Video Summarizer & Transcriber').classes('text-xl text-gray-400 font-light')

    # Upload Section
    with ui.column().classes('w-full max-w-2xl') as container_upload:
        with ui.card().classes('w-full glass-card p-10 items-center text-center'):
            ui.icon('cloud_upload', size='5rem', color='gray-400').classes('mb-4')
            ui.label('Drag & Drop your lecture video here').classes('text-xl font-bold mb-2')
            ui.label('Supports MP4, AVI, MKV up to 500MB').classes('text-sm text-gray-500 mb-6')
            
            ui.upload(
                label="Select Video File",
                auto_upload=True,
                max_file_size=500 * 1024 * 1024,
                on_upload=handle_upload
            ).props('color=blue-600 flat bordered accept=".mp4,.avi,.mov,.mkv,.webm"').classes('w-full max-w-sm')

    # Processing Section
    container_processing()

    # Results Section
    with ui.column().classes('w-full') as result_holder:
        result_holder.set_visibility(False) # Initially hidden
        container_results()

ui.run(title='Sidecut AI', port=8080, dark=True)
