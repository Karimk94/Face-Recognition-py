from werkzeug.serving import run_simple
from app import app
import os

if __name__ == '__main__':
    run_simple(
        '127.0.0.1',
        5001,
        app,
        use_reloader=True,
        use_debugger=True,
        threaded=True,
        exclude_patterns=[os.path.join(os.path.dirname(__file__), 'known_faces_db', '*')]
    )
