from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()
app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def main():
    if request.method == "POST":
        userInput = request.json['userInput']

        

        return jsonify({'message': f'POST request handled with data: {userInput}'})


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found', 'message': 'The requested URL was not found on the server.'}), 404


if __name__ == '__main__':
    app.run(debug=True)
