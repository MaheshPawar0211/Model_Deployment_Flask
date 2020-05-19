from flask import Flask, abort, jsonify, request
import _pickle as pickle
my_random_forest= pickle.load(open("iris_rc.pkl","rb"))
app=Flask(__name__)


@app.route('/api',methods=['POST'])
def make_predict():
    data=request.get_json(force=True)
    predict_request1=[data['sl'],data['sw'],data['pl'],data['pw']]
    print(predict_request1)
    y_hat = my_random_forest.predict([predict_request1])
    output=y_hat[0]
    print(output)
    return jsonify(results=int(output))

if __name__=='__main__':
    app.run(port=9000,debug=True)
