import json
from flask import Flask, send_from_directory, request

from seaconfig import *

app = Flask(__name__, static_folder='src')

@app.route('/')
def index():
    # send local file "index.html" as response
    return send_from_directory('.', 'index.html')

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/src/<path:path>')
def send_js(path):
    return send_from_directory('src', path)

@app.route('/p/<path:path>')
def send_p(path):
    return send_from_directory('p', path)

@app.route("/api/start_acquisition", methods=["POST"])
def start_acquisition():
    print("hello there")
    # get post data with key "config_file"
    json_data=None
    try:
        json_data=request.get_json()
    except Exception as e:
        pass

    if json_data is None:
        return json.dumps({"status": "error", "message": "no json data"})
    
    config = AcquisitionConfig.from_json(json_data["config_file"])

    return json.dumps({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
