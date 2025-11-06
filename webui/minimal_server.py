from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World - Kronos Web UI is working!'

@app.route('/api/test')
def test_api():
    return {'status': 'success', 'message': 'API is working'}

if __name__ == '__main__':
    print("Starting minimal test server on port 7070...")
    app.run(host='0.0.0.0', port=7070, debug=False)