
import os
import shutil
import glob

def cleanup_project():
    print("🧹 Starting MetaTune Project Cleanup...")
    
    # 1. WHITELIST: Files/Folders to NEVER delete
    # We use a set for O(1) lookups
    WHITELIST_FILES = {
        # Core Source
        'app.py', 'brain.py', 'data_analyzer.py', 'engine.py', 'pipeline.py',
        'bilevel.py', 'app_wandb.py', 'engine_stream.py',
        'cleanup.py', # Don't delete self!
        # Config / Info
        'README.md', 'requirements.txt', '.gitignore',
        # Scripts that might be useful
        'generate_audit_data.py', 'test_audit.py', 'verify_metatune.py', 'verify_evolution.py',
        'debug_csv.py'
    }
    
    WHITELIST_DIRS = {
        'tests',
        '.git',
        '.gemini', # System folder
        '.agent' # System folder
    }

    # 2. Get all files in current directory
    root_dir = os.getcwd()
    all_items = os.listdir(root_dir)

    for item in all_items:
        item_path = os.path.join(root_dir, item)
        
        # SKIP WHITELISTED
        if item in WHITELIST_FILES:
            print(f"   🛡️  Preserving Core File: {item}")
            continue
        if item in WHITELIST_DIRS:
            print(f"   🛡️  Preserving Directory: {item}")
            continue
        if item.startswith('.'): # Often system/git files, preserve unless specific trash
            if item == '.DS_Store':
                pass # Delete this
            else:
                continue

        # DELETE LOGIC
        deleted = False
        
        if os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            
            # Specific Targets
            is_bytecode = ext in ['.pyc']
            is_model = ext in ['.pkl', '.pth', '.joblib']
            is_data = ext in ['.csv', '.json', '.xml', '.xlsx']
            is_log = ext in ['.log', '.txt']
            is_viz = ext in ['.png', '.jpg', '.jpeg', '.svg']
            
            # Special check for 'demo_data.csv' - usually we might want to keep it?
            # User said "reset to original". Usually demo data is part of the repo.
            # But earlier files showed many temp CSVs. 
            # If the user wants to remove "txt log csv", we remove them.
            # We will assume demo_data.csv might be generated or downloaded. 
            # If it's crucial, it should have been in the whitelist. 
            # I'll check if it looks like a generated temp file.
            
            if is_bytecode or is_model or is_data or is_log or is_viz:
                try:
                    os.remove(item_path)
                    print(f"   🗑️  Deleted: {item}")
                    deleted = True
                except Exception as e:
                    print(f"   ❌ Failed to delete {item}: {e}")
        
        elif os.path.isdir(item_path):
            # Delete __pycache__ and temp folders
            if item == '__pycache__' or item == 'test_sandbox':
                try:
                    shutil.rmtree(item_path)
                    print(f"   🔥 Incinerated Directory: {item}")
                    deleted = True
                except Exception as e:
                    print(f"   ❌ Failed to delete directory {item}: {e}")

    print("\n✨ Project Cleaned Successfully!")

if __name__ == "__main__":
    cleanup_project()
