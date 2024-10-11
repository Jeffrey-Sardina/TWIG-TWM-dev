# adopted from https://www.delftstack.com/howto/python-flask/flask-display-image/
# and also from: https://www.geeksforgeeks.org/retrieving-html-from-data-using-flask/
# and https://www.w3schools.com/tags/tag_select.asp

# TWIG imports
from twm_draw import do_twm, load_hyps_dict

# external imports
from flask import Flask, request, render_template
import os
import sys

# Flask constructor
app = Flask(__name__) 

IMG_FOLDER = "static"
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
            pred_img = os.path.join(app.config["UPLOAD_FOLDER"], "twig-idea.png")
            true_img = os.path.join(app.config["UPLOAD_FOLDER"], "twig-nn.png")
            twm_text = f'Please fill in the whole form or else enjoy some LoTR memes!\n'
            twm_text += f'Specific errors follow:\n{err}'
            return render_template(
                "index.html",
                twm_pred_image=pred_img,
                twm_true_image=true_img,
                twm_text=twm_text,
            )
        assert False , "Not ready for newest TWIG version"
        img_pred, img_true, graph_save_url, mrr_pred, mrr_true, exp_id_wanted = do_twm(
            model_name="None",
            dataset_name=dataset,
            kge_model_name=kge_model_name,
            run_id=run_id,
            hyp_selected=hyp_selected,
            model_save_path=model_save_path,
            model_config_path=model_config_path,
            hyps_dict=hyps_dict,
            save_path=app.config["UPLOAD_FOLDER"],
            file_name_tag="twm"
        )
        twm_text = f'Predicted MRR (left image): {str(round(float(mrr_pred), 3)).ljust(5, "0")}<br>Ground Truth MRR (right image): {str(round(float(mrr_true), 3)).ljust(5, "0")}'
        hyp_text = f'Hyperparameter grid iD: {exp_id_wanted}. This maps to the hyperparameters:<br>'
        for key in hyp_selected:
            hyp_text += f' - {key} -> {hyp_selected[key]}<br>'
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
        twm_text = f'TWIG is cool!'
        return render_template(
            "index.html",
            twm_pred_image=img_pred,
            twm_true_image=img_true,
            twm_text=twm_text
        )
    
def start_twm_app(
        hyps_dict_path,
        kge_model_name_local,
        run_id_local,
        model_save_path_local,
        model_config_path_local
):
    global hyps_dict, kge_model_name, run_id, model_save_path, model_config_path
    hyps_dict = load_hyps_dict(hyps_dict_path)
    kge_model_name = kge_model_name_local
    run_id = run_id_local
    model_save_path = model_save_path_local
    model_config_path = model_config_path_local
    app.run(debug=True)
