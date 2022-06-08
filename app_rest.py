
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Only flask
from flask import Flask, request, render_template, make_response, jsonify
from flask_restplus import Resource, Api, fields
from gevent.pywsgi import WSGIServer

# DL packages
from mlib import beauty_gan as img_bt
import numpy as np

# Utility
from utils.util import base64_to_pil, np_to_base64_bt

# Flask and Flask_restplus declare
app = Flask(__name__)

# Swagger gui setting
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'

# Api initialize
api = Api(app, 
          # doc='/apidoc/',
          version='1.0', 
          title='Image DL API', 
          description='Images classfication and beauty GAN RestAPI'
    )

# Flask rest plus namespace define
predict_ns = api.namespace('predict', description='image prediction apis')

# Flask marshalling models
predict_ns_single_img = predict_ns.model(
    "Single input Image model",
    {
        "oriImage": fields.String(description="oriImage", required=True)
    },
)

predict_ns_two_img = predict_ns.model(
    "Two input Image model",
    {
        "oriImage": fields.String(description="oriImage", required=True),
        "mpImage": fields.String(description="mpImage", required=True)
    },
)

# Swagger Error Message
get_err_msg = { 
    200: 'Success',
    404: 'Page Not found',
}
post_err_msg = {
    200: 'Success',
    500: 'Wrong oriImg or mpImg parameters'
}

    
@api.route('/beauty')
class ImageBeautyIndex(Resource):
    @api.doc(responses=get_err_msg)
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index_beauty.html'), 200, headers)



@predict_ns.route('/img-beauty-single', methods=['POST'])
class ImageBeautyPredictSingle(Resource):
    @api.doc(responses=post_err_msg)
    @predict_ns.expect(predict_ns_single_img)
    def post(self):
        # Json request
        data = request.json
        
        # Image convert
        ori_img = base64_to_pil(data.get('oriImage'))
        
        # Test image save
        # ori_img.save("./image1.png")

        # Json response
        return jsonify(result=np_to_base64_bt(img_bt.predict_single_or_all(ori_img)))

@predict_ns.route('/img-beauty-all', methods=['POST'])
class ImageBeautyPredictAll(Resource):
    @api.doc(responses=post_err_msg)
    @predict_ns.expect(predict_ns_two_img)
    def post(self):
        # Json request
        data = request.json
        
        # Image convert
        ori_img, mp_img = base64_to_pil(data.get('oriImage')), base64_to_pil(data.get('mpImage'))
        
        # Test image save
        # ori_img.save("./image1.png")
        # mp_img.save("./image2.png")

        # Json response
        return jsonify(result=np_to_base64_bt(img_bt.predict_single_or_all(ori_img, mp_img)))

# RestApi errors handling
@predict_ns.errorhandler(Exception)
def predict_ns_handler(error):
    '''predict_ns error handler'''
    return {'message': 'Wrong oriImage or mpImage parameters or not face image.'}, getattr(error, 'code', 500)

if __name__ == '__main__':
    # Flask Server Start
    app.run(port=8080, threaded=False, debug=True) #디버그 모드 켜놨음 코드 수정하면 바로 반영된대

    # Gevent Server Start
    # Refer : https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/#gevent
    #http_server = WSGIServer(('127.0.0.1', 8080), app)

    # app.run('0,0,0,0', port=5000, debug=True)

    http_server = WSGIServer(('127.0.0.1', 8080), app)
    http_server.serve_forever()
