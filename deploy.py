from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer; 
import os, traceback
from dataset import create_text_data
from inference import inference
import json

from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def make_vocab_jsonable(vocab):
    # Convert Vocab object to a simple dictionary for JSON serialization
    return {token: vocab[token] for token in vocab.get_itos()}

def get_vocab():
    data_root_path = 'dataset'
    vocab_file = 'vocab.json'
    clean_text_dataset = f"{data_root_path}/text/Flickr8k.token.bahasa.clean.txt"
    tokenizer = get_tokenizer('basic_english')

    if os.path.exists(vocab_file):
        with open(vocab_file, 'r') as json_file:
            vocab = json.load(json_file)
    else:
        _, vocab = create_text_data(clean_text_dataset, tokenizer)
        # Convert the Vocab object to a JSON serializable dictionary
        jsonable_vocab = make_vocab_jsonable(vocab)
        with open(vocab_file, 'w') as json_file:
            json.dump(jsonable_vocab, json_file, indent=4)

    return vocab

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/coba')
def coba():
    return render_template('coba.html')

@app.route('/coba2')
def coba2():
    return render_template('coba2.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        d_model = 512
        n_head = 4

        # checkpoint_dir = f'checkpoints/{d_model}_{n_head}/'
        # model = checkpoint_dir + 'checkpoint_epoch_8.pth'
        model_s = f'saved_model/{d_model}_{n_head}_s.pth'
        model_m = f'saved_model/{d_model}_{n_head}_m.pth'
        
        cnn_backbone_s = 'efficientnetv2_rw_s.ra2_in1k'
        cnn_backbone_m = 'efficientnetv2_rw_m.agc_in1k'
        
        model = model_s
        cnn_backbone = cnn_backbone_s
        
        scale = request.form.get('scale')
        if scale == 'm':
            model = model_m
            cnn_backbone = cnn_backbone_m
            
        vocab = get_vocab()
    
        try:
            image = Image.open(filepath).convert('RGB')
            image.load()
            
            predicted_caption = inference(image=image, cnn_backbone=cnn_backbone, loaded_path=model, d_model=d_model, n_head=n_head, vocab=vocab)
            predicted_caption_ = ""
            for s in predicted_caption:
                predicted_caption_ += s + " "
                if s == "<end>":
                    x = predicted_caption_[:-9]
                    predicted_caption_ = predicted_caption_[:-9]
                    predicted_caption_ += "."
                    break    
            return jsonify({"caption": predicted_caption_}), 200
        except Exception as e:
            # Print the traceback to stderr (this goes to the console by default)
            traceback.print_exc()
            # You could also capture the traceback to a variable and return it in the response
            error_trace = traceback.format_exc()
            return jsonify({"error": str(e), "trace": error_trace}), 500
    return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True)

