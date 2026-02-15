import subprocess
from app.main import start

start()
subprocess.run(["streamlit","run","app/dashboard/streamlit_app.py"])
