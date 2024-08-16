# adopted from https://www.delftstack.com/howto/python-flask/flask-display-image/
# and also from: https://www.geeksforgeeks.org/retrieving-html-from-data-using-flask/
# and https://www.w3schools.com/tags/tag_select.asp

# importing Flask and other modules
from flask import Flask, request, render_template
import os
from twm_draw import do_twm, load_hyps_dict
import sys

# Flask constructor
app = Flask(__name__) 

IMG_FOLDER = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = IMG_FOLDER
hyp_selected = {}
hyps_dict = {}

# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def twm_demo():
    global hyp_selected
    if request.method == "POST":
        err = ""
        try:
            try:
                margin = request.form.get("margin")
                margin = float(margin)
                if request.form.get("loss") != "MarginRankingLoss":
                    err += "margin parameter can only be given when MarginRankingLoss is also given\n"
                    assert False, err
            except:
                if margin == 'None':
                    if request.form.get("loss") == "MarginRankingLoss":
                        err += "margin parameter must be set when MarginRankingLoss is given\n"
                        assert False, err
                    margin = None
            use_TWIG = request.form.get("use_twig")
            use_TWIG = True if use_TWIG == "Yes" else False
            dataset = request.form.get("dataset")
            hyp_selected = {
                "loss": request.form.get("loss"),
                "neg_samp": request.form.get("negsamp"),
                "lr":float( request.form.get("lr")),
                "reg_coeff": float(request.form.get("reg_coeff")),
                "npp": int(request.form.get("npp")),
                "margin": margin,
                "dim": int(request.form.get("dim"))
            }
        except:
            img = os.path.join(app.config["UPLOAD_FOLDER"], "temp.jpg")
            twm_text = f'Please fill in the whole form or else enjoy some LoTR memes!\n'
            twm_text += f'Specific errors follow:\n{err}'
            return render_template(
                "index.html",
                twm_image=img,
                twm_text=twm_text,
            )
        img_pred, img_true, graph_save_url, mrr_pred, mrr_true, exp_id_wanted = do_twm(
            dataset_name=dataset,
            kge_model_name=kge_model_name,
            run_id=run_id,
            hyp_selected=hyp_selected,
            model_save_path=model_save_path,
            model_config_path=model_config_path,
            hyps_dict=hyps_dict
        )
        img_pred = os.path.join(app.config["UPLOAD_FOLDER"], img_pred)
        img_true = os.path.join(app.config["UPLOAD_FOLDER"], img_true)
        twm_text = f'Predicted MRR: {str(round(float(mrr_pred), 3)).ljust(5, "0")}\nGround Truth MRR: {str(round(float(mrr_true), 3)).ljust(5, "0")}'
        hyp_text = f'Hyperparameter grid iD: {exp_id_wanted}'
        link_label = f'Graph URL accession: '
        link = f'{graph_save_url}'
        return render_template(
            "index.html",
            twm_pred_image=img_pred,
            twm_true_image=img_true,
            twm_text=twm_text,
            hyp_text=hyp_text,
            link_label=link_label,
            link=link
        )
    else:
        img_pred = os.path.join(app.config["UPLOAD_FOLDER"], "twig-idea.png")
        img_true = os.path.join(app.config["UPLOAD_FOLDER"], "twig-nn.png")
        twm_text = f'Enjoy some LoTR memes!'
        return render_template(
            "index.html",
            twm_pred_image=img_pred,
            twm_true_image=img_true,
            twm_text=twm_text
        )
    
if __name__ == '__main__':
    hyps_dict_path = sys.argv[1]
    kge_model_name = 'ComplEx'
    run_id = '2.1'
    model_save_path = ""
    model_config_path = ""
    hyps_dict = load_hyps_dict(hyps_dict_path)
    app.run(debug=True)
