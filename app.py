from flask import Flask, request, jsonify
from model.model import Detection

detection_model = Detection()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def process_image():
    file = request.files['image']
    file.save('im-received.jpg')
    # Read the image via file.stream
    pred = detection_model.predict('im-received.jpg')
    print(file.stream)
    print(pred)

    return jsonify({'msg': 'success'})


if __name__ == "__main__":
    app.run(debug=True)