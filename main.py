import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'taste-trove-credentials.json'
storage_client = storage.Client()

def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req

model = load_model('my_model_fix.h5', custom_objects={'req': req})

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'message': 'No file part in the request.'}), 400
            
            file = request.files['file']

            # Check if the file is a valid image
            try:
                img = Image.open(file)
                img.verify()  # Verify that it is, in fact, an image
                file.seek(0)  # Reset file pointer to the beginning after verify
            except (IOError, SyntaxError) as e:
                return jsonify({'message': 'File is not a valid image.', 'error': str(e)}), 400

            # Upload the file to Google Cloud Storage
            image_bucket = storage_client.get_bucket('taste_trove_bucket_model')
            img_blob = image_bucket.blob('predict_uploads/' + file.filename)
            img_blob.upload_from_file(file)

            # Re-open the file and convert it to a BytesIO object for processing
            file.seek(0)
            img_bytes = BytesIO(file.read())

            img = tf.keras.utils.load_img(img_bytes, target_size=(224, 224))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            images = np.vstack([x])

            # Model prediction
            pred_food = model.predict(images)
            maxx = pred_food.max()

            nama = ['Babi Guling', 
                    'Es Pisang Ijo',
                    'Kerak Telur',
                    'Rendang', 
                    'Sate Ulat Sagu', 
                    'Soto Banjar']
        
            lokasi = ['Bali', 
                      'Sulawesi',
                      'Jakarta',
                      'Sumatera Barat', 
                      'Papua', 
                      'Kalimantan Selatan']
            
            deskripsi = ['Babi guling atau babi putar (di Bali disebut be guling) adalah sejenis makanan yang terbuat dari anak babi betina atau jantan yang perutnya diisikan dengan bumbu dan sayuran seperti daun ketela pohon dan lalu dipanggang sambil diputar-putar (diguling-gulingkan) sampai matang dengan ditandai dengan perubahan warna kulit menjadi kecokelatan dan renyah',
                         'Es pisang ijo adalah kudapan khas dari Makassar, Sulawesi Selatan. Hidangan ini diolah dari buah pisang raja, ambon, atau kepok yang sudah matang. Pisang dibalut dengan adonan tepung beras bercampur santan dan air daun pandan yang memberi warna hijau dan aroma pandan.',
                         'Merupakan makanan khas Jakarta, kerak telor merupakan jajanan kebanggaan warga Jakarta, memiliki rasa yang gurih, kerak telor memanfaatkan kelapa sebagai bahan khas-nya dan kerap hadir di pesta dan perayaan kemerdekaan RIdan Ulang Tahun Kota Jakarta.',
                         'Rendang adalah salah satu masakan tradisional Minangkabau yang menggunakan daging dan santan kelapa sebagai bahan utama dengan kandungan bumbu yang kaya akan rempah-rempah.',
                         'Hidangan ini terbuat dari ulat sagu, yang ditemukan di dalam batang pohon sagu yang melimpah di hutan-hutan sekitar mereka. Ulat sagu ini bukan hanya merupakan sumber protein yang kaya, tetapi juga memiliki rasa yang unik.',
                         'Soto Banjar merupakan salah satu kuliner yang terkenal di Indonesia. Berbeda dengan soto lainnya di Indonesia, soto ini memiliki keunikan dalam penyajian dan cita rasanya. Soto Banjar ini menggunakan ayam kampung sebagai bahan utamanya. Bumbu yang di haluskan berupa bawang merah, bawang putih, merica.']
            
            

            if maxx <= 0.75:
                respond = jsonify({'message': 'Makanan tidak terdeteksi'})
                respond.status_code = 400
                return respond

            result = {
                "nama": nama[np.argmax(pred_food)],
               
                "lokasi": lokasi[np.argmax(pred_food)],

                "deskripsi": deskripsi[np.argmax(pred_food)]
                
            }

            respond = jsonify(result)
            respond.status_code = 200
            return respond

        except Exception as e:
            respond = jsonify({'message': 'Error processing image file', 'error': str(e)})
            respond.status_code = 400
            return respond

    return 'OK'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
