from os import listdir, remove
from os.path import isfile, join, splitext
import time
import face_recognition
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import requests

# Global storage for images
faces_dict = {}
persistent_faces = "/root/faces"

# Create flask app
app = Flask(__name__)
CORS(app)

# <Picture functions> #


def is_picture(filename):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions


def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]


def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

def calc_face_encoding(image):
    # Currently only use first face found on picture
    loaded_image = face_recognition.load_image_file(image)
    faces = face_recognition.face_encodings(loaded_image)

    # If more than one face on the given image was found -> error
    if len(faces) > 1:
        raise Exception(
            "Found more than one face in the given training image.")

    # If none face on the given image was found -> error
    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]


def get_faces_dict(path):
    image_files = get_all_picture_files(path)
    return dict([(remove_file_ext(image), calc_face_encoding(image))
        for image in image_files])


def detect_faces_in_image(file_stream):
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)

    # Get face encodings for any faces in the uploaded image
    uploaded_faces = face_recognition.face_encodings(img)

    # Defaults for the result object
    faces_found = len(uploaded_faces)
    faces = []

    if faces_found:
        face_encodings = list(faces_dict.values())
        for uploaded_face in uploaded_faces:
            match_results = face_recognition.compare_faces(
                face_encodings, uploaded_face)
            for idx, match in enumerate(match_results):
                if match:
                    match = list(faces_dict.keys())[idx]
                    match_encoding = face_encodings[idx]
                    dist = face_recognition.face_distance([match_encoding],
                            uploaded_face)[0]
                    faces.append({
                        "id": match,
                        "dist": dist
                    })

    return {
        "count": faces_found,
        "faces": faces
    }
def compare_faces_in_image(file_stream, face_id):
    tsStart = time.time()
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    tsDownload = time.time()
    # Get face encodings for any faces in the uploaded image
    uploaded_faces = face_recognition.face_encodings(img)
    tsEncoding = time.time()
    # Defaults for the result object
    faces = []

    if len(uploaded_faces):
        
        filename = "{0}/{1}.jpg".format(persistent_faces, request.args.get('id'))
        face_encodings = []
        face_encodings.append(calc_face_encoding(filename))
        ts = time.time()
        dist = face_recognition.face_distance(face_encodings, uploaded_faces[0])[0]
        faces.append({
            "id": face_id,
            "dist": dist,
            "processTime": time.time() - ts,
            "totalTime": time.time() - tsStart,
            "donwloadTime": tsDownload - tsStart,
            "encodingTime": tsEncoding - tsDownload
        })
        print(jsonify(faces))

    return {
        "faces": faces
    }
# <Picture functions> #

# <Controller>


@app.route('/', methods=['POST'])
def web_recognize():
    file = extract_image(request)

    if file and is_picture(file.filename):
        # The image file seems valid! Detect faces and return the result.
        return jsonify(detect_faces_in_image(file))
    else:
        raise BadRequest("Given file is invalid!")
        
@app.route('/compare', methods=['POST'])
def web_compare():
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")
    # file = extract_image(request)

    # if file and is_picture(file.filename):
        # The image file seems valid! Detect faces and return the result.
    return jsonify({
        "faces": [{
        "id": request.args['id'],
        "dist": 0.1,
        "processTime": 0,
        "totalTime": 0,
        "donwloadTime": 0,
        "encodingTime": 0
    }]
    })
        # return jsonify(compare_faces_in_image(file, request.args.get('id')))
    # else:
        # raise BadRequest("Given file is invalid!")


@app.route('/faces', methods=['GET', 'POST', 'DELETE'])
def web_faces():
    # GET
    if request.method == 'GET':
        return jsonify(list(faces_dict.keys()))

    # POST/DELETE
    file = extract_image(request)
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")

    if request.method == 'POST':
        app.logger.info('%s loaded', file.filename)
        # HINT jpg included just for the image check -> this is faster then passing boolean var through few methods
        # TODO add method for extension persistence - do not forget abut the deletion
        file.save("{0}/{1}.jpg".format(persistent_faces, request.args.get('id')))
        try:
            #new_encoding = calc_face_encoding(file)
            faces_dict.update({request.args.get('id'): 1})
        except Exception as exception:
            raise BadRequest(exception)

    elif request.method == 'DELETE':
        remove("{0}/{1}.jpg".format(persistent_faces, request.args.get('id')))

    return jsonify(list(faces_dict.keys()))

@app.route('/face', methods=['GET'])
def url_faces():
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")
    if 'url' not in request.args:
        raise BadRequest("Identifier for the face was not given!")

    if request.method == 'GET':
        url = request.args.get('url')
        app.logger.info('%s loaded', url)
        # HINT jpg included just for the image check -> this is faster then passing boolean var through few methods
        # TODO add method for extension persistence - do not forget abut the deletion
        
        file = requests.get(url)
        filename = "{0}/{1}.jpg".format(persistent_faces, request.args.get('id'))
        open(filename, "wb").write(file.content)
        #new_encoding = calc_face_encoding(filename)
        faces_dict.update({request.args.get('id'): 1})
    return jsonify(list(faces_dict.keys()))

def extract_image(request):
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")

    return file
# </Controller>


if __name__ == "__main__":
    print("Starting by generating encodings for found images...")
    # Calculate known faces
    # faces_dict = get_faces_dict(persistent_faces)
    # print(faces_dict)

    # Start app
    print("Starting WebServer...")
    app.run(host='0.0.0.0', port=8080, debug=False)
