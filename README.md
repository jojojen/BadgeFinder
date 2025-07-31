# BadgeFinder

# venv setting
conda activate py312
source env/bin/activate
pip install -r requirements.txt

# exec method 1
python app.py
# exec method 2
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
# exec method 3
export FLASK_APP=app.py
flask run

# check at
http://127.0.0.1:5000/