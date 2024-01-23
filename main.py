from dotenv import load_dotenv
from flask import Flask, jsonify, request

from services.chatbot import master_chatbot
from services.initializeServer import initialize_server

load_dotenv()
app = Flask(__name__)


@app.before_first_request
def before_first_request():
    initialize_server()


@app.route("/api/", methods=["POST"])
def main():
    if request.method == "POST":
        userMessage = request.json["userMessage"]

        chat_res = master_chatbot(userMessage)

        return jsonify({"message": chat_res})


@app.errorhandler(404)
def not_found_error(error):
    return (
        jsonify(
            {
                "error": "Not Found",
                "message": "The requested URL was not found on the server.",
            }
        ),
        404,
    )


if __name__ == "__main__":
    app.run(debug=True)
