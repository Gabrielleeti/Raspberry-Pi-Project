from flask import Flask, render_template, jsonify

app = Flask(__name__)

names = []  # List to store names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_name/<name>')
def add_name(name):
    names.append(name)
    return jsonify(success=True)

@app.route('/get_names')
def get_names():
    return jsonify(names=names)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 80,debug = True)