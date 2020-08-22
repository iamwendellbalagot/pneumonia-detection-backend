from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model import Detection
import base64
import io

detection_model = Detection()

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def process_image():
    #file = request.files['image']
    #file.save('im-received.jpg')
    #pred = detection_model.predict('im-received.jpg')
    #print(file.stream)
    #print(pred)
    payload = request.get_json()
    print('PAYLOAD: ',type(payload['image']))
    im_b64 = payload['image']
    im_binary = base64.b64decode(im_b64)
    #buf = io.BytesIO(im_binary)
    
    result = float(detection_model.predict(im_binary))
    result = {'msg': 'success', 'prediction': result}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)